#include <torch/extension.h>
#include <vector>

using torch::Tensor;

static Tensor put(const Tensor& base, const Tensor& rows,
                  const Tensor& columns, const Tensor& values) {
    c10::List<c10::optional<Tensor>> indices;
    indices.push_back(rows);
    indices.push_back(columns);
    return base.index_put(indices, values, false);
}

// These are ATen calls, including their ordinary autograd nodes. No analytic
// replacement VJP, changed reduction, fused multiply-add or CUDA atomics here.
std::vector<Tensor> events(
    const Tensor& ids, const Tensor& valid, const Tensor& rays,
    const Tensor& pixels, const Tensor& centers, const Tensor& conics,
    const Tensor& opacities, const Tensor& normals, const Tensor& distance,
    double extent, double near, double far) {
    TORCH_CHECK(centers.scalar_type() == torch::kFloat32 && valid.scalar_type() == torch::kBool,
                "original float32/valid slot inputs required");
    const int64_t B = ids.size(0), N = ids.size(1), P = pixels.size(1);
    Tensor rows, slots, columns, compact_ids, compact_valid;
    int64_t K;
    {
        torch::NoGradGuard guard;
        auto xy = centers.index({ids});
        auto q = conics.index({ids});
        auto a = q.select(-2, 0).select(-1, 0);
        auto b = (q.select(-2, 0).select(-1, 1) + q.select(-2, 1).select(-1, 0)) * .5;
        auto c = q.select(-2, 1).select(-1, 1);
        auto det = a * c - b.square();
        auto safe = (a > 0) & (c > 0) & (det > 0)
                    & torch::isfinite(q).all(-1).all(-1) & torch::isfinite(xy).all(-1);
        auto denominator = torch::where(safe, det, torch::ones_like(det));
        auto rx = extent * (torch::where(safe, c, torch::ones_like(c)) / denominator).sqrt();
        auto ry = extent * (torch::where(safe, a, torch::ones_like(a)) / denominator).sqrt();
        safe = safe & torch::isfinite(rx) & torch::isfinite(ry);
        auto low = pixels.amin({1}), high = pixels.amax({1});
        auto overlap = (xy.select(-1, 0) + rx + 1. >= low.select(-1, 0).unsqueeze(1))
                     & (xy.select(-1, 0) - rx - 1. <= high.select(-1, 0).unsqueeze(1))
                     & (xy.select(-1, 1) + ry + 1. >= low.select(-1, 1).unsqueeze(1))
                     & (xy.select(-1, 1) - ry - 1. <= high.select(-1, 1).unsqueeze(1));
        auto keep = valid & (~safe | overlap);
        K = keep.sum(1).max().item<int64_t>();
        auto indices = keep.nonzero();
        rows = indices.select(1, 0);
        slots = indices.select(1, 1);
        auto rank = keep.to(torch::kLong).cumsum(1) - 1;
        columns = rank.index({rows, slots});
        compact_ids = torch::zeros({B, K}, ids.options());
        compact_valid = torch::zeros({B, K}, valid.options());
        compact_ids.index_put_({rows, columns}, ids.index({rows, slots}));
        compact_valid.index_put_({rows, columns}, true);
    }
    auto pixel_normals = normals.index({ids}).unsqueeze(2).expand({-1, -1, P, -1});
    if (K == 0) {
        auto zero = torch::zeros({B, N, P}, pixels.options()) + centers.sum() * 0;
        return {zero, zero, pixel_normals};
    }
    auto original_delta = pixels.unsqueeze(1) - centers.index({ids}).unsqueeze(2);
    auto delta = put(torch::zeros({B, K, P, 2}, original_delta.options()), rows, columns,
                     original_delta.index({rows, slots}));
    auto squared = at::einsum("tlpi,tlij,tlpj->tlp", {delta, conics.index({compact_ids}), delta});
    auto response = torch::exp(-.5 * squared);
    auto normal = normals.index({compact_ids}).unsqueeze(2).expand({-1, -1, P, -1});
    auto denominator = (normal * rays.unsqueeze(1)).sum(-1);
    auto safe = torch::where(denominator.abs() > 1e-8, denominator, torch::ones_like(denominator));
    auto depth = distance.index({compact_ids}).unsqueeze(2) / safe;
    response = torch::where(denominator.abs() > 1e-8, response, 0);
    response = torch::where(compact_valid.unsqueeze(2) & (squared <= extent * extent)
                           & (depth > near) & (depth < far), response, 0);
    auto alpha = opacities.index({compact_ids}).unsqueeze(2) * response;
    auto restored_alpha = put(torch::zeros({B, N, P}, alpha.options()), rows, slots,
                              alpha.index({rows, columns}));
    auto restored_depth = put(torch::zeros({B, N, P}, depth.options()), rows, slots,
                              depth.index({rows, columns}));
    return {restored_alpha, restored_depth, pixel_normals};
}

std::vector<Tensor> composite(
    const Tensor& input_alpha, const Tensor& colors, const Tensor& depths,
    const Tensor& normals, const std::string& distortions, bool detach_l1) {
    auto alphas = input_alpha.clamp(0, .999);
    auto prefix = torch::cat({torch::ones_like(alphas.slice(1, 0, 1)), 1 - alphas}, 1).cumprod(1);
    auto weights = alphas * prefix.slice(1, 0, -1);
    auto alpha = weights.sum(1);
    auto rgb = at::einsum("bnp,bnc->bpc", {weights, colors});
    auto znum = (weights * depths).sum(1);
    auto nnum = (weights.unsqueeze(-1) * normals).sum(1);
    auto w = weights.detach();
    auto l1 = torch::zeros_like(alpha);
    if (distortions == "l1" || distortions == "both") {
        auto sorted = depths.sort(1);
        auto sorted_depth = std::get<0>(sorted), order = std::get<1>(sorted);
        auto sorted_w = (detach_l1 ? w : weights).gather(1, order);
        auto pw = sorted_w.cumsum(1) - sorted_w;
        auto pwd = (sorted_w * sorted_depth).cumsum(1) - sorted_w * sorted_depth;
        l1 = (2 * sorted_w * (sorted_depth * pw - pwd)).sum(1).clamp_min(0);
    }
    auto l2 = torch::zeros_like(alpha);
    if (distortions == "l2" || distortions == "both")
        l2 = (2 * (w.sum(1) * (w * depths.square()).sum(1)
                    - (w * depths).sum(1).square())).clamp_min(0);
    Tensor median;
    if (depths.size(1)) {
        auto crossed = weights.cumsum(1) >= .5;
        auto index = crossed.to(torch::kLong).argmax(1, true);
        median = torch::where(crossed.any(1), depths.gather(1, index).squeeze(1), 0);
    } else median = torch::zeros_like(alpha);
    return {rgb, alpha, znum, nnum, l1, l2, median, weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("events", &events);
    m.def("composite", &composite);
}
