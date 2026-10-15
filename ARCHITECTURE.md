# SplatForge 架构设计

> 状态：Architecture Freeze Candidate  
> 日期：2026-09-12  
> 适用范围：SplatForge v0.x 的核心抽象、依赖方向、算子契约、实现提供者、质量治理与发布边界  
> 本文不是某个客户重建流程的设计，也不是 PGSR、Gaussian Wrapping 或 PaMO 的复刻计划。

## 0. 结论先行

SplatForge 应被定义为：

> 一个 PyTorch-first、面向高性能与可微计算、允许商业使用的空间算子系统。Gaussian Splatting 是第一个领域扩展，不是整个系统的基础对象。

这意味着它既不是“另一套 3DGS 训练框架”，也不是“装着十几个论文 CUDA 子模块的仓库”。真正需要长期稳定的是一条很窄的腰部：

~~~text
版本化的数学算子契约
        +
基于张量元数据的实现选择
        +
可审计、可替换的实现提供者
~~~

第一版冻结以下原则：

1. 核心 runtime 不认识 Gaussian、PointCloud、Mesh、Robot、Scene 或完整训练模型。
2. 不设计万能 <code>Primitive</code>、万能 <code>SceneObject</code> 或万能 <code>Renderer</code>。
3. 扩展性沿三条互相独立的轴发生：Operator、Representation、Implementation Provider。
4. 公共算子只接收完成该计算所需的最小张量和少量静态参数。
5. Representation view 只存在于 API 边界；进入 kernel 前必须显式 lowering 成张量。
6. 数学语义与实现分离：<code>OpSpec</code> 定义“结果是什么”，<code>ImplementationManifest</code> 定义“在哪里、怎样运行”。
7. plain PyTorch reference、Triton 和 native CUDA 是同一语义的不同 provider，不是三套 API。
8. 允许 Gaussian 专属算子与高价值 fused op；禁止它们反向污染 runtime。
9. 自动选择只能发生在语义等价的实现之间；算法变化必须由用户显式选择。
10. 不允许 silent fallback、silent cast、silent device transfer、silent contiguous copy 或 silent detach。
11. 规范、reference、conformance tests、容差和性能基线不能由实现者在同一个变更里一起修改。
12. 未实际运行的支持矩阵格子必须显示为 untested，不能由相邻平台推断。
13. 基础包在没有 CUDA extension、编译器或 NVIDIA GPU 时仍能 import，并能运行已声明的 Torch reference。
14. v0.x 只承诺 Python 语义 API；不承诺 C++ ABI、CUDA kernel ABI 或序列化 graph ABI。
15. 自有代码采用 Apache-2.0；许可证和来源追踪按文件、算子和发行物执行。

目录只是这些边界在文件系统上的投影。即使目录看起来整齐，只要依赖可以反向穿透，架构仍然是失败的。

---

## 1. 项目使命与边界

### 1.1 要解决的问题

3DV/CG 论文代码常见的演化方式是：每篇方法 fork 一份 rasterizer，增加几个输出、改一套 backward、再绑定一个完整训练模型。结果往往包括：

- 数学语义隐藏在 CUDA 临时 buffer 和训练代码中；
- depth、normal、visibility、patch matching 等能力互不兼容；
- 同一个阶段被多个项目重复实现；
- 许可证沿 fork 链变得不适合宽松商业集成；
- 只能通过整个论文仓库调用一个本来很通用的算子；
- 正确性、梯度、性能和支持平台没有独立证据。

SplatForge 的价值不是收集这些代码，而是把可复用部分重新定义成：

- 有明确数学语义的算子；
- 有高精度 reference 的验收对象；
- 有可替换的 Torch、Triton、CUDA 实现；
- 有真实支持矩阵和发布证据；
- 不依赖某篇论文的训练对象、数据集或目录结构。

### 1.2 初始领域

初始工作量来自 Gaussian geometry、投影、rasterization、patch warp、NCC/ZNCC、visibility 和 mesh extraction 前后的几何计算。这只决定 v0.x 的算子优先级，不决定核心 ontology。

PGSR、Gaussian Wrapping、PaMO 或其他论文工程可以作为：

- 行为需求来源；
- 数学与性能对照；
- 外部 integration / example；
- 端到端验收消费者。

它们不能成为：

- runtime 依赖；
- 公共类型体系；
- 算子命名空间；
- 许可证来源不明的代码来源；
- SplatForge 是否能独立安装的前提。

视频、SAM、COLMAP、MapAnything、训练入口、客户任务编排、最终 mesh 修复与交付属于单独的应用或 pipeline 仓库。SplatForge 可以被它调用，但不能反向依赖它。

### 1.3 明确不做

v0.x 不做：

- 通用训练框架；
- 数据集管理器；
- 通用场景图；
- 自己的 Tensor/Device runtime；
- 统一一切 3D 表示的继承树；
- 自动猜测第三方模型结构；
- 通用 shader language；
- 自定义计算图 IR；
- 自动分布式调度；
- 插件市场；
- 稳定 C++ ABI；
- 为尚未出现的 ROCm、Metal、机器人中间件提前建复杂抽象；
- 对完整论文方法做“可商用”担保。

### 1.4 成功标准

SplatForge 的架构成功，不以文件数、算子数或 README 长度判断，而以这些事实判断：

- 新增 point-cloud 算子时不修改 Gaussian 或 runtime；
- 新增 mesh 表示时不增加“空 covariance、空 opacity”等伪字段；
- 新增 CUDA provider 时不修改公开函数签名和数学规范；
- 移除全部原生扩展后，基础包仍可安装、import、运行 reference；
- 一个算子的 Torch/Triton/CUDA 实现运行同一套 conformance suite；
- 用户可以查询实际选择了哪个实现以及其他实现为何被拒绝；
- 每个发布声明都能追溯到具体矩阵格子和原始证据；
- 另一个 LLM 只能改实现文件，不能通过修改考卷来宣布完成。

---

## 2. 三条正交扩展轴

不要把“算法”“设备”和“kernel 写法”都叫 backend。

| 轴 | 解决的问题 | 示例 | 谁选择 |
|---|---|---|---|
| Operator / algorithm variant | 计算的数学语义是什么 | pinhole projection、EWA projection、expected depth、median depth | 公共 API 或显式静态参数 |
| Representation | 用户的数据怎样映射到算子输入 | point set、Gaussian geometry、triangle mesh、ray bundle | view / adapter / recipe |
| Implementation provider | 同一语义怎样高效执行 | plain Torch、Triton、native CUDA | dispatch policy |

设备是执行签名的一部分，例如 CPU、CUDA、以后可能的 ROCm/MPS；它与 provider 仍不是同一个概念。同一张 CUDA 卡上可能同时存在 Torch composition、Triton 和 native CUDA 三个等价实现。

关键约束：

- provider 只能改变执行方式，不能改变数学语义；
- 近似算法、不同排序规则或不同梯度不能伪装成“更快 provider”；
- representation 只负责显式映射，不决定 kernel；
- 新 operator 在现有契约系统内注册，不要求 runtime 理解它所属的领域。

---

## 3. 依赖方向：真正的薄腰

~~~mermaid
flowchart TB
    APP[Applications / Training / Paper Reproductions]
    INT[Explicit Integrations and Adapters]
    REC[Domain Facades and Semantic Recipes]
    OPS[Versioned Semantic Operators]
    CORE[Contracts, Registry, Dispatch, Execution Policy]
    TORCH[Torch Provider]
    TRITON[Triton Provider]
    CUDA[Native CUDA Provider]

    APP --> INT
    INT --> REC
    REC --> OPS
    OPS --> CORE
    TORCH -. registers implementation .-> CORE
    TRITON -. registers implementation .-> CORE
    CUDA -. registers implementation .-> CORE
~~~

### 3.1 允许的依赖

| 逻辑层 | 可以依赖 | 禁止依赖 |
|---|---|---|
| contracts/runtime | PyTorch 张量元数据、标准库、schema 工具 | domain、representation、provider、论文项目 |
| semantic operator API | contracts/runtime、共享 conventions | provider、integration、完整 model |
| domain view/facade | operator API、自身的 view/facet | provider 私有 kernel、其他领域内部对象 |
| provider | contracts、它实现的 OpSpec、PyTorch/Triton/CUDA | integration、训练器、其他 provider |
| integration | 第三方项目、view、recipe、operator | 无需成为其他层依赖 |
| application | 以上公开层 | 不得被 SplatForge 核心反向 import |

Backend/provider 是向 registry 注入实现，不是 semantic API 直接 import 并硬编码调用的模块。

### 3.2 核心 runtime 只认识什么

核心可以认识：

- <code>torch.Tensor</code>；
- shape、stride、dtype、device、layout；
- requires-grad mask；
- 静态 attributes；
- <code>OpKey</code> 与 semantic major；
- <code>ExecutionPolicy</code>；
- 输出 metadata；
- workspace requirement；
- implementation manifest；
- 当前 device、stream 和 build fingerprint。

核心不得出现：

- <code>GaussianModel</code>；
- <code>PointCloud</code>；
- <code>TriangleMesh</code>；
- <code>Scene</code>；
- optimizer、densification statistics 或 checkpoint；
- PGSR、Gaussian Wrapping、PaMO 等项目名；
- 数据集路径、固定相机数量、固定分辨率或固定 GPU 型号。

### 3.3 自动执行的 architecture fitness rules

以下规则不能只写在文档里，必须由 CI 检查：

1. AST/import graph 检查禁止反向依赖和循环依赖。
2. runtime/core 路径禁止 import domain、provider 与 integrations。
3. 公共 operator signature 禁止接收项目 model 或未审查的任意对象。
4. domain 代码禁止直接调用 <code>_C</code>、Triton kernel 或 provider 私有符号。
5. examples、benchmarks、datasets 不得成为 runtime dependency。
6. 新增 provider 的 diff 不得触碰 public API、OpSpec 或其他 provider。
7. 新增非 GS 垂直切片时不得修改 Gaussian 源码。
8. <code>import splatforge</code> 不得触发 JIT 编译或加载全部插件。

---

## 4. 数据抽象：窄 view 与 facet，不是万能 Primitive

### 4.1 为什么拒绝 Primitive 继承树

下面这种设计看似通用，实际会把所有领域耦合在一起：

~~~python
class Primitive:
    position
    scale = None
    rotation = None
    opacity = None
    normal = None
    faces = None
    robot_link = None
~~~

它会导致：

- optional field 越来越多；
- 点云被迫伪装成“退化 Gaussian”；
- mesh 和 Gaussian 为同名字段争夺语义；
- backend 根据对象类型和缺失字段做动态分支；
- 新领域必须修改旧基类；
- 训练状态很快渗入 kernel API。

正确做法是轻量、零拷贝、无独立 storage/cache 的 view/facet 组合。它们表达张量语义，并持有底层 Tensor 引用，但不复制数据、不拥有另一份存储，也不承载训练状态。

### 4.2 冻结 view 原则，不提前冻结万能 facet 类型

v0.x 冻结“轻量零拷贝、无独立 storage/cache、显式 lowering、无继承”的原则，但下面的类型只作为设计示例，不立即承诺成 stable API。

~~~python
@dataclass(frozen=True)
class PointSetView:
    positions: Tensor
    batching: Batching
    frame: FrameTag
    unit: LengthUnit

@dataclass(frozen=True)
class GaussianCovarianceView:
    means: Tensor
    covariances: Tensor
    batching: Batching
    frame: FrameTag
    unit: LengthUnit

@dataclass(frozen=True)
class GaussianScaleRotationView:
    means: Tensor
    scales: Tensor
    rotations: Tensor
    batching: Batching
    frame: FrameTag
    unit: LengthUnit
    quaternion_order: QuaternionOrder

@dataclass(frozen=True)
class TriangleMeshView:
    vertices: Tensor
    faces: Tensor
    batching: MeshBatching
    frame: FrameTag
    unit: LengthUnit
~~~

<code>Batching</code> 是显式 sum type，而不是用 <code>offsets=None</code> 暗示多种情况：

~~~text
Unbatched
DenseBatch(batch_size)
PaddedBatch(lengths 或 mask)
PackedBatch(offsets)
~~~

batching 由 collection view 唯一拥有，不能让每个 field 各自携带可能互相矛盾的 offsets。Mesh 使用专门的 <code>MeshBatching(vertex_offsets, face_offsets, index_scope)</code>，分别表达 vertex/face 分段和 local/global index scope。

不同 operand arity 使用不同 view 和不同 operator signature。例如 full covariance 是一个 operand，scale + rotation 是两个 operands；不能用一个 <code>CovarianceEncoding</code> 枚举偷偷改变输入数量。

FrameTag、IndexScope 等可扩展标识不得做成每加一个领域就必须修改 core 的 closed enum。它们可以是 namespaced、可校验的静态值；真正进入稳定 API 前由第二个实际用例验证。

<code>@dataclass(frozen=True)</code> 只禁止 wrapper 字段重新绑定，不保证内部 Tensor 深层不可变。真正的 mutation/aliasing 仍由每个 operator contract 和 PyTorch schema 约束。

这些 view 只做：

- 边界验证；
- frame、unit、association 和 batch 语义检查；
- 将明确的字段 lowering 为张量和静态 attributes；
- 给高层 recipe 提供可读接口。

它们绝不：

- 在 GPU 热路径中产生逐元素动态多态；
- 保存 optimizer、梯度统计、densification 状态；
- 自动寻找字符串字段；
- 隐式移动设备、改变 dtype、materialize 大型中间量；
- 成为第三方模型的基类。

typed facade 可以在边界检查 frame/unit；直接调用 raw tensor operator 的用户则承担满足规范的责任。两个数值 tensor 来自不同 frame 时，runtime 无法凭数值自动发现，文档和类型层不能做虚假保证。

只有当两个真实领域共享完全相同的语义时，才抽出可复用 facet。v0.x 不建立任意字段 registry。

### 4.3 Gaussian 的数学表示

SplatForge 的 Gaussian 层表达数学 Gaussian，而不是原版 3DGS 的训练参数对象：

~~~text
中心 means
协方差 covariance 或明确编码的 covariance factor
alpha / weight
任意 per-element payload
~~~

以下是参数化或训练策略，不是核心 primitive 字段：

- scale + quaternion；
- opacity logits；
- spherical-harmonic coefficients；
- densification statistics；
- optimizer state。

它们通过显式算子转换：

~~~python
covariance = sf.gaussian.covariance_from_scale_rotation(
    scales,
    rotations,
    quaternion_order="wxyz",
)

features = sf.appearance.evaluate_sh(coefficients, view_directions)
~~~

如果 materialize covariance 太慢，应定义一个拥有自己 OpSpec 的显式 fused semantic op，例如：

~~~text
gaussian.project_covariance@1(covariance, ...)
gaussian.project_scale_rotation@1(scales, rotations, ...)
~~~

第二个 op 的 reference 可以定义为“转换 + 投影”，native CUDA 再实现等价融合。普通 Python recipe 不会被 provider 自动识别，也不能反过来规定所有空间对象都必须携带 scale 和 quaternion。

### 4.4 Camera 也不用万能基类

采用具体数学模型和明确命名：

- <code>PinholeCameraView</code>；
- <code>OrthographicCameraView</code>；
- <code>FisheyeCameraView</code>；
- <code>world_to_camera</code> 或 <code>camera_to_world</code>，不使用含糊的 <code>extrinsics</code>。

对应算子可以是：

- <code>camera.project_pinhole@1</code>；
- <code>camera.project_orthographic@1</code>；
- <code>camera.generate_rays_pinhole@1</code>。

下游若需要共用，只共用 UV/depth/ray 等数学结果，不在 CUDA kernel 内做 camera virtual callback。

### 4.5 变长 batch 是契约

统一承认三种形式，但每个算子必须明确自己支持哪些：

~~~text
dense/exact   [B, N, ...]
padded        [B, Nmax, ...] + lengths 或 mask
packed        [sum(Ni), ...] + offsets[B+1]
~~~

禁止根据 tensor rank 猜 batch 语义，禁止偷偷把 packed pad 成 dense。

packed contract 必须定义并测试：

- 空集合；
- 空 segment；
- 单元素；
- 极大 segment；
- offsets 单调性和边界；
- int32/int64 索引范围和溢出行为。

### 4.6 v0.x 不公开加速结构 handle

BVH、hash grid、tile index 和邻域索引在 v0.x 保持 provider-private execution resource。这样纯 tensor operator 才能在 Torch/Triton/CUDA 间做可验证选择，fake/export 也不会被一个不透明、绑定 provider 的对象破坏。

不要建立 <code>GaussianBVH</code>、<code>PointBVH</code>、<code>RobotBVH</code> 三套公共类型。

未来确有多个外部消费者需要显式 build/query/refit 时，再通过独立 RFC 设计 Resource Plane。它必须与纯 tensor operator 分开，并明确：

- handle 绑定的 provider、device 和 build fingerprint；
- query 不允许 auto 切换到不兼容 provider；
- ownership、mutation、alias、析构和 stream lifetime；
- source generation 和 refit 规则；
- serialization/versioning；
- 默认不承诺 compile/export。

---

## 5. Operator Contract：什么叫“这个算子”

### 5.1 唯一权威 OpSpec

每个公开算子只有一份机器可读 manifest 和一份人类可读数学说明。Torch、Triton、CUDA 与文档都从同一个契约出发。

示例：

~~~yaml
id: camera.project_pinhole@1
maturity: candidate
owners:
  - geometry-maintainers
spec_package: specs/camera/project_pinhole/

operands:
  - name: points
    shape: [..., N, 3]
    dtype_family: floating
    device_relation: common
    differentiable: true
  - name: world_to_camera
    shape: [..., 4, 4]
    dtype_relation: points
    differentiable: true
  - name: intrinsics
    shape: [..., 3, 3]
    dtype_relation: points
    differentiable: true

attributes:
  - name: image_size
    type: tuple[int, int]
  - name: pixel_center
    enum: [integer, half]

outputs:
  - name: uv
    shape: [..., N, 2]
  - name: depth
    shape: [..., N]
  - name: valid
    shape: [..., N]
    dtype: bool

semantics:
  transform_direction: world_to_camera
  handedness: right_handed
  depth_axis: positive_z
  mutation: none
  aliasing: none
  empty_input: return_empty_outputs
  invalid_depth: valid_false
  nan_inf: propagate_and_mark_invalid
  dtype_promotion: explicit_table
  output_cardinality: fixed

autograd:
  vjp: [points, world_to_camera, intrinsics]
  jvp: not_yet_supported
  second_order: not_yet_supported
  undefined_boundaries:
    - depth_equals_zero

numerics:
  accumulator_dtype: per_dtype_profile
  profiles: [strict, balanced, fast]
  tolerance_table: tests/tolerances/camera.project_pinhole.yaml

evidence:
  conformance_suite: tests/conformance/camera/project_pinhole
  benchmark_suite: benchmarks/camera/project_pinhole
  provenance: provenance/camera.project_pinhole.yaml
~~~

身份规则：

- <code>OpKey = namespace.name@semantic_major</code>；
- <code>spec_revision</code> 是规范包规范化后的不可变 content hash，由 catalog/evidence 工具计算并存放在包外索引中，避免自引用 hash；
- provider 和可信 evidence 都绑定精确 spec revision；
- 任何可观察语义或 schema 变化使用新 major 或新 OpKey；
- 不给 operator 再维护一套会被 dispatch 忽略的 minor/patch SemVer。

“机器 manifest”和“数学 prose”不是两个可以互相矛盾的权威来源。一个 OpSpec 是一个规范包：结构化 schema、数学定义和必需表格共同参与 canonical digest；重复字段只保存一处，README/API 文档从它生成。

上面的 <code>shape: [..., N, 3]</code> 目前只是说明语法。Phase 0 必须先定义 named-axis、broadcast、packed overload 的正式 validator；在此之前使用 typed Python validator，不能宣称这段伪 DSL 已能自动生成全部检查。

### 5.2 契约必须覆盖的内容

以下都属于 API，而不是 CUDA 作者自行决定的细节：

- 公式及参考文献；
- tensor operand 与 static attribute 的区分；
- rank、symbolic shape 和 broadcasting；
- dtype promotion 与 accumulator dtype；
- device、layout、stride 和 non-contiguous 行为；
- dense/padded/packed batch；
- coordinate frame、单位、handedness；
- world-to-camera / camera-to-world；
- quaternion 顺序；
- 像素中心、NDC 范围；
- mutation、aliasing 与输出分配；
- 空输入、NaN/Inf、奇异矩阵和退化几何；
- ordering、tie-breaking、clipping 和 visibility；
- 输出 cardinality；
- deterministic 语义；
- 每个输入的 VJP/JVP/高阶梯度能力；
- 不可导边界采用的约定；
- 错误类型和错误发生时机；
- 每个 dtype/case class 的容差。

默认规则：

- functional；
- 不修改输入；
- 不返回未声明 alias；
- 不静默转换；
- 不静默复制；
- 不静默同步 host；
- 不支持就结构化报错。

性能需要 <code>out=</code> 时，把它设计成独立契约，不做“有时分配、有时原地写”的 maybe-out。

### 5.3 输出语义与分配策略分开

OpSpec 只定义可观察语义：

~~~yaml
output:
  cardinality: fixed | data_dependent
  layout: dense | padded | packed
  public_count_overflow_semantics: none
~~~

如果调用者必须提供 capacity，并能观察 count/overflow，这就是不同的 public schema/overload 或 OpKey，而不是某个 provider 偷偷选择的分配策略。

ImplementationManifest/ExecutionPlan 再声明执行方式：

~~~yaml
allocation_protocol: inferred | two_pass | provider_private
host_synchronization: none | required
execution:
  eager: supported
  compile: supported | unsupported
  export: supported | unsupported
  cuda_graph: supported | unsupported
~~~

例如 radius search 的语义可以是 data-dependent + packed；Torch provider 使用 two-pass/host-sync，另一个 provider 可能使用不同私有策略。host synchronization、workspace 和 capture 能力必须出现在 support/explain_dispatch 中，不强迫所有 provider 使用同一实现协议。

不要假装动态邻域或 mesh 输出第一天就支持 export/CUDA Graph；不支持时显式标记。

### 5.4 哪些东西才值得成为 public operator

至少满足以下条件中的大部分：

1. 有独立、可解释的数学语义；
2. 输入输出不暴露 provider 私有临时布局；
3. 值得独立测试或存在多个实现；
4. 被多个 recipe 使用，或本身就是用户需要的完整计算；
5. 可以承诺版本兼容。

类似 <code>duplicateWithKeys</code>、<code>identifyTileRanges</code>、<code>renderCUDABackwardV4</code> 的内部阶段应保持 provider-private。

---

## 6. Implementation Provider 与 Dispatch

### 6.1 OpSpec 和实现 manifest 分开

<code>OpSpec</code> 决定“什么结果才正确”。实现 manifest 只说明“这个实现何时有资格运行”。

~~~yaml
implementation_id: native_cuda.camera.project_pinhole.sm80.v1
provider: native_cuda
provider_api: 1
op: camera.project_pinhole@1
op_spec_revision: "sha256:<exact-spec-digest>"
implementation_version: 1.3.2
lowering_target: splatforge_native::project_pinhole_v1
static_variant_token: "<immutable artifact-local token>"

declared_eligibility:
  device: cuda
  compute_capability: ">= 8.0"
  dtypes: [float16, bfloat16, float32]
  layouts:
    - contiguous
    - strided_last_dim
  backward: true
  backward_may_copy_grad_output: false
  jvp: false
  determinism_guarantee: same_build_same_device
  autocast: explicit_dtype_table
  cuda_graph: true
  dynamic_shapes: bounded
  host_synchronization: none
  precision_profiles: [balanced, fast]
  workspace: query_at_plan_time

build:
  source_revision: ...
  wheel_tag: ...
  python_abi: ...
  os_cpu_arch: ...
  libc_baseline: ...
  torch_version_and_build_flavor: ...
  libtorch_abi_mode: regular | stable
  libtorch_stable_abi_target: ...
  cxx11_abi: ...
  cuda_build_toolkit: ...
  cuda_runtime_and_linked_libraries: [...]
  minimum_driver: ...
  sass_targets: [...]
  ptx_target_and_version: ...
  compiler_fingerprint: ...
~~~

manifest 中的 declared eligibility 只是 provider 作者声明的候选过滤条件，不等于“已验证支持”。verified support 只能由可信 CI 证据产生，release tier 只能由维护者批准。

<code>lowering_target + static_variant_token + provider-library digest</code> 必须唯一定位实际可调用实现。若某个 provider op 在内部自行选择 variant，那么这些内部路径就不能再被包装成不同 implementation IDs 或分别声称 capability；二者只能选一种身份模型。

例如 <code>compute_capability &gt;= 8.0</code> 只表示算法层 declared eligibility，不代表当前二进制可加载或可执行。实际 binary eligibility 还要匹配 Python/wheel ABI、OS/libc、PyTorch build/ABI、CUDA/driver，以及 wheel 中真实嵌入的 SASS/PTX targets。arch list 必须由 release 配置显式固定，不能依赖构建机当前可见 GPU。

一个从未实测的新 GPU 代际即使满足算法条件、也有可 JIT 的 PTX，仍是 untested/best-effort，不能自动继承 Tier 1。

实现的 <code>support(call_signature, policy)</code> 必须返回结构化结果，包括拒绝原因：

- <code>unsupported_dtype</code>；
- <code>unsupported_layout</code>；
- <code>no_backward</code>；
- <code>cuda_arch_too_old</code>；
- <code>binary_abi_incompatible</code>；
- <code>driver_or_ptx_incompatible</code>；
- <code>not_graph_safe</code>；
- <code>workspace_limit_exceeded</code>；
- <code>precision_policy_mismatch</code>。

### 6.2 Dispatch 只读取 metadata

<code>CallSignature</code> 包含：

~~~text
OpKey + exact spec revision
input device / dtype / rank / shape / stride
static attributes
requires_grad mask + grad-enabled/inference/forward-AD mode
device capability
effective autocast / determinism / precision / copies / host-sync policy
definition + implementation registry generation
verification-catalog digest/generation
~~~

public wrapper 在当前 package/runtime snapshot 中固定绑定一个 definition handle（OpKey + exact spec revision）；<code>supports</code>、<code>plan</code>、<code>explain_dispatch</code> 和实际调用必须使用同一个 handle，绝不在调用时寻找“latest revision”。

dispatch 不读取 tensor 内容。否则可能触发 GPU 到 CPU 同步、破坏 compile/capture、造成数据依赖的性能与不可复现选择。

选择与执行流程：

1. 校验 schema；
2. 规范化 static attributes；
3. 在不把 SymInt 强制转成 Python int/bool 的前提下构造 <code>CallSignature</code>；
4. 枚举该 OpKey + exact spec revision 的 implementations；
5. 先用 trusted availability/advisory catalog 排除 quarantined/disabled cell；
6. 再用 declared eligibility、verified evidence requirement 和 execution policy 过滤；
7. 按稳定规则选择具体 implementation；
8. 建立该 implementation 的 allocation/workspace/synchronization 执行协议；
9. fixed output 由 metadata 推导；data-dependent output 按该 implementation 声明的协议执行；caller-capacity 只有在被调用的是显式 capacity schema 时才成立；
10. 只缓存不依赖 tensor 内容的 plan 部分，cache key 包含 registry/catalog generations 与全部有效 policy；
11. 在当前 device/current stream 上执行。

data-dependent count 阶段不是 provider selection；其 count/overflow 结果若公开才属于 OpSpec，two-pass 或 host-sync 本身属于 implementation plan。只有在 PyTorch 对应版本能准确表达并经过测试时才声明 compile/export；否则明确 eager-only。

### 6.3 ExecutionPolicy

v0.x 只暴露少量跨 provider、语义清楚的策略：

~~~python
policy = sf.ExecutionPolicy(
    providers=("native_cuda", "triton", "torch"),
    runtime_failure="error",
    copies="forbid",
    host_sync="forbid",
    precision="balanced",
    determinism="inherit",       # inherit | require | allow
    autocast="inherit",          # inherit | forbid | explicit
    evidence_requirement="release_verified",
    allow_unverified=False,
    require_cuda_graph=False,
)
~~~

公开 API 不出现 <code>tile_size=16</code>、<code>warp_reduce_v3=True</code> 等实现旋钮。它们属于 provider 内部 variant、autotune 或 benchmark 配置。

规则：

- 默认用户可以使用 <code>auto</code>；
- 测试、benchmark 和生产验收固定 implementation；
- 显式 <code>provider="native_cuda"</code> 不支持时必须报错；
- <code>auto</code> 可以在授权且语义等价的候选中正常选择，这不叫 fallback；
- Torch reference 或已证明的 decomposition 若与 OpSpec 等价，应注册为该 OpKey 的普通 implementation，并明确出现在 provider order；没有合格 candidate 时一律报错；
- kernel launch/runtime failure 后 v0.x 一律报错，不自动换实现重试；kernel 可能已部分写入输出，重试既不安全也不一定语义等价；
- approximate method 不能由 provider policy 偷偷选择；
- 不同 OpKey 之间的替代算法只能由 domain facade 或用户显式选择；
- 默认 <code>auto</code> 只选择随安装包携带的可信 capability catalog 中满足当前 release policy、标为 conformant 的精确 cell；
- experimental/untested implementation 只能用精确 implementation ID 加 <code>allow_unverified=True</code> 显式选择；加载外部 provider 只授权执行其代码，不授权它成为默认数值实现；
- quarantined/disabled cell 永远从公共 dispatch 排除，显式选择也报错；
- <code>torch.use_deterministic_algorithms(True)</code> 与 SplatForge policy 取更严格要求；没有满足保证的 provider 就报错；
- autocast 状态、effective input dtype、determinism requirement 和 precision profile 都进入 CallSignature、plan guard 与 evidence。

必须提供：

~~~python
sf.supports("camera.project_pinhole@1", inputs, policy=policy)
sf.explain_dispatch("camera.project_pinhole@1", inputs, policy=policy)
plan = sf.plan("camera.project_pinhole@1", inputs, policy=policy)
~~~

<code>explain_dispatch</code> 至少返回实际选择、所有候选、拒绝原因、verified evidence 状态、适用 advisory/quarantine、发生的显式转换和 build fingerprint。

### 6.4 Definition Registry、Implementation Registry 和插件加载

不要把规范与实现塞进一个可随意修改的全局表。

<code>OpDefinitionRegistry</code>：

- 接受新的 namespaced OpKey；
- 一个 runtime snapshot 内，每个 OpKey 只映射一个不可变 spec revision；
- 同一 OpKey 只有 digest 完全相同才允许幂等注册，其他情况立即冲突；
- 规范澄清通过新的 package/release snapshot 产生新 revision；语义变化产生新 major；
- 每次事务式添加后递增 definition generation。

<code>ImplementationRegistry</code>：

- 允许用户在运行后显式、事务式加载新 provider；
- 每次变化递增 implementation generation 并使相应 dispatch cache 失效；
- 拒绝重复 implementation ID；
- 禁止 monkeypatch 覆盖旧实现；
- 校验 provider API 与精确 spec revision；
- 保持确定的选择顺序；
- 可导出 declared eligibility 和 verified capability matrix。

runtime 另有不可由 provider 自行覆盖的 effective-availability gate。采用无环的两层身份：

1. wheel 内的 signed build/capability manifest 绑定 provider shared-library digest、implementation ID 与 build fingerprint，不包含 enclosing wheel 自己的 hash；
2. wheel 外的 detached signed release manifest 以最终 wheel SHA-256 为键，映射到内置 manifest digest、capability/advisory catalog 和 release evidence。

安装后的 runtime 用内置签名 manifest 校验实际 shared library；release 审计用 detached manifest 校验最终 wheel。availability gate 可以按 <code>implementation ID + library/artifact digest + CallSignature predicate</code> 禁用特定 GPU/dtype/grad 格子，而不必粗暴禁用全部实现。

已经创建的 <code>ExecutionPlan</code> 固定原 implementation、spec revision 和 registry generation，不因随后加载 provider 而改变。生产环境如需完全固定，可显式调用 <code>runtime.seal()</code>；不能以“第一次 plan 自动冻结”阻止正常的延迟加载。

两个 registry 都应支持独立实例，方便测试和嵌入。

第三方 provider 必须由用户显式加载。以后可用 Python entry points 做发现，但“发现”不能等于 import 并执行任意已安装代码：

~~~python
sf.plugins.discover()
sf.plugins.load("vendor.cuda_provider")
~~~

所谓“事务式加载”只覆盖同一进程里的 SplatForge registries。Python import、native <code>dlopen</code> 和 <code>torch.library</code> 全局注册可能不可回滚，也不能从一个验证进程“搬运”到另一个进程，因此采用进程级提交：

1. 在执行插件代码前读取静态 manifest，预检 namespace、ID 预留、provider API、spec digest、binary ABI 和许可；
2. 可先在 disposable validation worker 中完整 load + smoke test；其注册只属于该 worker，不向父进程转移；
3. 生产 supervisor 启动新的 serving worker，在该 worker 内重新 load/register，并只在同一进程的 PyTorch 注册全部成功后提交该 worker 的 SplatForge registry；
4. worker 报告 sealed snapshot digest 和 health check 通过后，supervisor 才把请求路由给它。

如果 native/PyTorch 全局注册只完成一半就失败，该 worker/process 标记为 tainted 并销毁，不能宣称回滚成功。单进程开发模式发生此错误时必须重启。生产环境在启动阶段加载批准插件，随后 <code>runtime.seal()</code>。

### 6.5 PyTorch 集成边界

v0.x 的 Python/frontend 数据货币是 <code>torch.Tensor</code>，不另造 Tensor runtime。这里必须区分三层：

- 数学 OpSpec 尽量与框架无关；
- v0.x host/frontend 和 PyTorch registration 使用 <code>torch.Tensor</code>；
- native CUDA device kernel 使用私有的 pointer、shape、stride 与 POD 参数，不把 ATen/Python 对象带进核心计算。

ATen/PyTorch glue 与 CUDA device code 保持薄而分离。这不是现在承诺 C++ ABI，而是避免未来增加 C++/其他 frontend 时必须重写数学 kernel。

- 可由 PyTorch primitives 清楚表达的 reference 保持普通 Python composition；
- Triton 实现通过适合 <code>torch.compile</code> 追踪的 PyTorch/Triton registration 接入；
- native C++/CUDA 使用 PyTorch custom operator registration；
- 每个承诺 compile/export 的稳定 custom op 提供准确 fake/meta kernel，匹配 shape、dtype、device、layout、stride、storage offset 和适用的 alias metadata；
- autograd 使用正式 registration，把 forward/backward 作为能力 bundle；
- <code>torch.library.opcheck</code> 用于检查注册正确性，但不能替代数值和梯度测试；
- 公共 API 不暴露 <code>torch.autograd.Function</code> 类、<code>torch.ops.splatforge_native.*</code> 或 <code>_C.*</code>。

SplatForge provider selection 位于 PyTorch device dispatcher 之上。native CUDA 与 Triton 不能被当成同一个 CUDA dispatch key 下的两个候选，期待 PyTorch 自动择优；PyTorch dispatch key 主要解决设备/layout 等分派。

每个 provider 使用不同的私有 operator identity：

~~~text
splatforge_native::project_pinhole_v1
splatforge_triton::project_pinhole_v1
~~~

公共 wrapper 先选择具体 provider，再 lowering 到对应私有 op。fake/meta/autograd registration 绑定这个具体 op；这样才能独立测试、固定、quarantine 和记录 build。

provider identity 仍不够：一个 provider 内可能有多个 implementation variant。ExecutionPlan 必须绑定精确 implementation ID；lowering 要么调用该 implementation 独有的私有 op identity，要么传入不可变的 static variant token。provider 私有 op 收到 plan 后不得再次按 tensor 内容或环境自动重选。exact implementation ID/token 必须进入 compiled graph guard、backward context、evidence 和 export manifest。

概念上的边界：

~~~python
# 稳定语义 API
sf.camera.project_pinhole(...)

# 私有 native binding，不向用户承诺
torch.ops.splatforge_native.project_pinhole_v1(...)
~~~

<code>torch.compile/export</code> 的执行模型必须固定：

1. v0.x 的 compile/export 要求一个显式 <code>runtime.seal()</code> 产生的不可变 RuntimeSnapshot；
2. snapshot token 包含 definition、implementation、capability/quarantine catalog、policy 和 execution-mode digests/generations；
3. public Python wrapper 在 eager/trace 阶段实际读取这个 token，使 Dynamo 对它建立 guard，而不是只在文档里声称有 generation；
4. wrapper 只根据 tensor metadata、static attrs 和 sealed policy 解析 provider；
5. 选择结果 lowering 成 plain Torch composition、Triton op 或一个绑定 exact implementation 的私有 custom op；
6. compiled graph 中保留固定的具体实现，不在 graph 内遍历 Python registry；
7. Dynamo 同时 guard dtype、shape、stride、static attrs、grad/inference/autocast/determinism mode 和 snapshot token；
8. active snapshot/catalog 变化会替换 token并触发下一次调用 retrace，不原地改变旧 graph；
9. <code>sf.plan(...).run()</code> 在 v0.x 只用于 graph 外预规划；
10. export artifact 记录 snapshot digest、OpKey、spec revision、provider、implementation ID 和 build digest；除非另有 portable lowering，不宣称可跨 provider 执行，也不能假装后来的 quarantine 能远程改写离线 artifact。

compile CI 必须从公开 <code>sf.*</code> wrapper 进入，而不只测试私有 <code>torch.ops.*</code>。v0.x 的 compiled/export 稳定层只返回 Tensor 或固定 tuple；若以后公开 dataclass result，必须先注册并测试 PyTree 行为。

autocast 不依赖“环境开着就会自动正确”。每个 lowering target 明确声明并实现：

- no-cast；
- 针对 CPU/CUDA 的显式 autocast registration 和 dtype table；
- 或经过证明的 fallthrough。

wrapper 不得在 manifest 之外偷偷 cast。effective input/output/accumulator dtype、采用的 autocast rule 和任何允许的 copy/cast 都进入 explain、guard 和 evidence。

### 6.6 Autograd 在选择之前检查

provider 注册 forward/backward bundle：

- forward launcher；
- context setup；
- 需要保存的最小 tensor；
- VJP；
- 可选 JVP；
- backward workspace；
- 高阶梯度能力。

dispatch 使用有效 differentiation mode，而不只看 Tensor 的 <code>requires_grad</code>：<code>no_grad</code>、inference mode、forward AD 和 autocast 都进入签名。只有当前调用实际需要 backward/JVP 时才要求对应能力。

一旦选择 forward implementation，其配套 backward/context strategy 也随 ExecutionPlan 固定；backward 不能再次运行一般 provider dispatch。不能 forward 跑完才发现无法反传，也不能 silent detach。

training-capable implementation 必须处理 PyTorch 可能传入的任意合法 upstream-gradient stride（包括 non-contiguous/expanded 情形）。若 backward 需要 materialize contiguous gradient，manifest 必须声明 <code>backward_may_copy_grad_output</code>，forward dispatch 在 <code>copies="forbid"</code> 时不能选择它；copy 的时间和显存进入 benchmark/evidence。不能到 backward 才悄悄复制或报一个规划阶段无法预见的支持错误。

v0.x 的训练路径只使用 functional private op。它返回 fresh semantic outputs，加上 schema 固定、仅供 autograd context 使用的 auxiliary outputs；公共 wrapper 可以隐藏 auxiliary，但私有 schema、fake kernel 和 tests 必须包含它们。这样 <code>register_autograd</code> 的 context setup 能从输入/输出保存 backward 所需状态。

mutable <code>out/workspace</code> private op 在 v0.x 明确 <code>backward=false</code>，只用于 inference/预规划路径。若未来训练必须使用 mutable buffer，需独立 RFC 并采用正确的 C++ Autograd dispatch-key 设计，不能把 non-functional schema 硬塞给 Python <code>register_autograd</code>。

### 6.7 Workspace、allocator、stream

禁止 provider 保留隐式全局 scratch tensor。

~~~python
with torch.inference_mode():
    plan = sf.plan(op, inputs, policy=policy)
    workspace = sf.allocate_workspace(plan)
    result = plan.run(inputs, workspace=workspace)
~~~

ExecutionPlan 记录：

- implementation ID；
- 输出 metadata；
- forward/backward workspace 大小与 alignment；
- 是否需要清零；
- 有效 shape 范围；
- graph-capture 声明；
- build fingerprint。

公共和训练用 private functional op 必须返回 fresh、互不产生未声明 alias 的 Tensor。若 inference-only 私有 custom op 接收预分配 output/workspace Tensor 并写入，PyTorch schema 必须准确声明所有 mutation/aliasing；不能用外层“看起来 functional”隐藏对输入参数的写操作。

需要 <code>out</code> 或 workspace 写入时采用独立、标记 non-differentiable 的私有 schema。conformance 同时验证数值与 mutation contract。

CUDA provider 必须：

- 使用当前 PyTorch CUDA stream；
- 使用正确 device guard；
- 不固定 default stream；
- 不调用隐式 <code>cudaDeviceSynchronize</code>；
- 不通过读取 GPU scalar 到 host 选择路径；
- 正确处理多 GPU；
- 对跨 stream workspace 复用使用 event 管理生命周期。

ExecutionPolicy 使用 context-local 状态，不使用可被任意线程修改的进程全局变量。

---

## 7. 组合与融合：只融合显式算子

普通 Python recipe 只是组合；v0.x 没有自定义 IR、pattern rewrite 或图编译器，因此 provider 看不到并自动替换任意上游 composition。

例如一个可读 recipe 可以是：

~~~text
decode covariance
  → project means
  → project covariance
  → compute screen bounds
  → bin/order
  → coverage
  → alpha composite
~~~

如果该组合确实需要融合，必须将它提升为一个显式 fused semantic operator，拥有自己的 OpKey、OpSpec、reference、gradient contract、provider manifest 和 conformance suite。例如：

~~~text
gaussian.project_covariance@1(covariance, camera...)
gaussian.project_scale_rotation@1(scales, rotations, camera...)
~~~

第二个算子的 reference 可以组合 covariance decode 与 projection；native CUDA 可以为第二个算子实现 fused kernel。因为两个算子的输入和梯度对象不同，它们不能伪装成同一实现。

显式 fused op 必须满足：

- 公共可观察输出等价；
- gradient contract 等价；
- 精度 profile 合格；
- 中间私有 buffer 不进入 API；
- implementation manifest 如实声明支持范围。

不承诺：

- 自动识别任意 Python recipe；
- provider 为普通函数调用序列直接注册融合；
- 在没有独立 OpSpec 时跨算子改变执行。

只有满足这些条件的高频组合才升级为 public fused op：

- 真实 workload 中反复出现；
- fusion 有显著性能或显存收益；
- 语义稳定；
- 能被独立 conformance suite 验证。

“第二个真实用例才抽取，第三个独立消费者才稳定”作为默认规则：

- 只被 Gaussian 使用的概念先留在 Gaussian domain；
- Point 也出现完全相同的数学阶段后再抽到共享 op；
- 第三个调用方出现后再承诺 stable API。

---

## 8. Gaussian / Splat 层的具体设计

### 8.1 共享什么，不共享什么

正确的复用：

~~~text
Gaussian 3D
  ├─ project means
  ├─ project covariance
  └─ screen-space ellipse ─┐
                           ├─ ellipse coverage / bin / composite
Point splat                │
  ├─ project points        │
  └─ isotropic footprint ──┘

Triangle mesh
  └─ triangle rasterization / ray intersection
     语义不同，不强塞进 ellipse rasterizer
~~~

候选共享数学算子：

- point/vector transform；
- pinhole projection 与 ray generation；
- covariance transform；
- radius/KNN search；
- spatial hashing；
- segmented sort/reduction；
- screen-space ellipse coverage；
- ordered alpha compositing；
- ray-AABB、ray-triangle；
- mesh adjacency。

Gaussian-specific operator 完全可以存在。真正禁止的是 Gaussian-specific ontology 进入 runtime。

### 8.2 建议的初始 semantic operator

~~~python
uv, depth, valid = sf.camera.project_pinhole(
    points,
    world_to_camera,
    intrinsics,
    image_size=(height, width),
)

covariance = sf.gaussian.covariance_from_scale_rotation(
    scales,
    rotations,
    quaternion_order="wxyz",
)

ellipse, depth, valid = sf.gaussian.project_covariance(
    means,
    covariance,
    world_to_camera,
    intrinsics,
    image_size=(height, width),
)

patch_a, valid_a = sf.sampling.patch_warp(image_a, sampling_grid_a)
patch_b, valid_b = sf.sampling.patch_warp(image_b, sampling_grid_b)

scores, valid = sf.matching.patch_zncc(
    patch_a,
    patch_b,
    sample_mask=requested_mask & valid_a & valid_b,
)
~~~

Warp 与 ZNCC 是两个契约：前者拥有插值、像素中心、越界和 grid gradient，后者只拥有已经采样的 patch reduction、mask、低方差与 reduction gradient。论文消费者的固定 <code>7×7</code> 是 provider/profile 支持单元，不写进通用 ZNCC ontology。

Point cloud 可以只使用 <code>camera.project_pinhole</code>，不需要 covariance 或 opacity。Mesh 使用自己的 <code>mesh.*</code>，不需要伪装成 splat。

### 8.3 避免第二个“巨型 rasterization 函数”

论文 fork 常见的问题不是 output 多，而是把四种不同变化混进同一个参数列表：

1. camera/sensor 模型；
2. 数学算法变体；
3. 想要的输出统计；
4. provider 和性能旋钮。

SplatForge 把它们分开：

- camera 先生成明确的投影输入；
- 算法变体使用不同 op 或显式 semantic attribute；
- compiled thin-waist API 使用固定 tensor operands 和固定 output schema；
- provider policy 不进入数学参数。

建议区分两类输出：

**线性 payload 聚合**

- 任意 per-element feature；
- 使用同一组 alpha/compositing 权重做 weighted accumulation；
- color、learned feature、未归一化 normal 都可复用这一通道。

**非线性或结构化统计**

- alpha/transmittance；
- expected depth；
- transmittance quantile / median depth；
- first-hit 或 visibility；
- per-element contribution/count；
- argmax、排序或诊断统计。

后一类每项都需要独立语义、边界行为和梯度契约，不能塞进不稳定的 <code>meta: Dict</code>，也不能把 tensor 藏在“静态 request”对象里。

v0.x 使用少数固定 schema 的 operator：

~~~python
accumulated, alpha = sf.splat.rasterize_features(
    ellipse,
    depth,
    opacity,
    payload,                 # [..., N, C]，显式 tensor operand
    image_size=(height, width),
)

alpha, depth_num, normal_num = sf.splat.rasterize_geometry(
    ellipse,
    depth,
    opacity,
    normals,                 # 显式 tensor operand
    image_size=(height, width),
)

median_depth, valid = sf.splat.rasterize_depth_quantile(
    ellipse,
    depth,
    opacity,
    q=0.5,
    image_size=(height, width),
)

contribution = sf.splat.rasterize_contribution(
    ellipse,
    depth,
    opacity,
    image_size=(height, width),
)
~~~

规则：

- 任意线性 per-element 属性沿最后的 <code>C</code> 维拼接，一次聚合后由调用者显式切片；
- expected feature/depth 等派生结果由版本化 recipe 根据 numerator、alpha 和明确的 epsilon/invalid 规则计算；
- median/quantile、visibility、per-element contribution 等非线性统计使用独立 OpKey 或固定 canonical bundle；
- 每个 compiled op 返回固定 Tensor/tuple schema；eager facade 可以再包装成 dataclass，公开后须先注册并测试 PyTree；
- provider 可以共享 private bin/order/composite 代码，并为固定 op 选择内部模板 variant；
- 新 output 先形成独立 OpSpec/reference/tests，再决定独立执行还是建立显式 fused op；
- 与同一 OpSpec 等价且经过测试的 decomposition 注册为普通 implementation；其他 output/算法组合只能显式调用，不能自动猜分解；
- v0.x 不在稳定薄腰中提供动态 request builder。

可以在 eager-only 的 domain facade 提供方便接口，例如 <code>sf.gaussian.render(outputs=...)</code>。它负责验证和调用上述固定 operators，但必须明确标记不属于 compiled/export 稳定 API，并且不能把 payload 字典直接传进 kernel/dispatch。

这样新增 geometry output 不需要 fork 一整份训练 rasterizer，同时也不会用动态返回 schema 破坏 PyTorch graph，或把 mesh renderer 强行塞入同一个接口。

### 8.4 Public stage 与 private stage

建议稳定或候选的边界：

- <code>camera.project_pinhole</code>；
- <code>gaussian.covariance_from_scale_rotation</code>；
- <code>gaussian.project_covariance</code>；
- <code>matching.patch_warp</code>；
- <code>matching.patch_zncc</code>；
- <code>splat.rasterize_features</code>；
- 经过真实需求验证的固定 geometry/statistics raster ops；
- 将来经过需求验证的 <code>spatial.radius_search</code>；
- 将来经过需求验证的 <code>mesh.ray_triangle_intersect</code>。

tile duplication、tile range、临时排序 key、backward scratch 等先保持 private。若未来确实被多个算法独立使用，再按 Operator Proposal 晋级。

---

## 9. 外部表示、框架与插件

### 9.1 显式 adapter

首版使用：

~~~python
view = sf.integrations.gsplat.from_tensor_layout(
    means=means,
    quaternions=quaternions,
    scales=scales,
    layout_version="documented-gsplat-version",
)
~~~

而不是：

~~~python
view = sf.guess_representation(model)
~~~

显式 adapter 更容易调试、编译和审计，也不会因多个插件为同一 Python 类型注册转换而产生歧义。

adapter 必须针对明确的第三方包版本、真实类型或 tensor layout；不能假设第三方存在一个并不存在的 canonical <code>Model</code>。

integration 模块：

- 是可选依赖；
- 不从顶层 <code>__init__</code> 自动 import；
- 只负责数据映射和 recipe；
- 不修改第三方 model；
- 不成为 core 或 operator 的依赖。

integration 另分发行等级：

- <code>permissive-default</code>：可进入默认 extras、wheel/test image，依赖许可证已通过发行审计；
- <code>restricted-external</code>：只记录兼容协议或放在独立仓库，不进入 SplatForge 默认 extra、wheel、测试镜像或“commercial-clean”支持声明。

PGSR 和 Gaussian Wrapping 的官方许可包含非商业限制，应属于 restricted-external。PaMO 的 AGPL-3.0 不是“禁止商业”，但其 copyleft 义务不符合 SplatForge 宽松核心目标，也不进入默认发行物。Apache-2.0 编写的 adapter 不会自动改变被调用项目的许可证义务。

未来 Open3D、Warp、JAX 等数据互操作优先通过 DLPack/array protocol 和窄 adapter，不重造第二套 Tensor。零拷贝数据互操作不等于跨框架自动保留 autograd；可微路径需要框架专用 bridge，并明确 stream、gradient buffer ownership 和生命周期。

### 9.2 外部 provider

外部 provider 需要：

- 唯一 namespace；
- provider API version；
- 明确实现的 OpKey + semantic major；
- support constraints；
- 显式加载；
- 冲突拒绝；
- conformance suite；
- provenance 与许可证记录。

插件加载即执行代码，因此不能自动信任环境中任意 entry point。

### 9.3 架构扩展验收

准备一个独立的外部测试包，在不修改 SplatForge runtime 和 Gaussian 源码的情况下完成：

1. 定义一个非 Gaussian operator；
2. 提供 Torch reference；
3. 注册一个 provider；
4. 运行公共 conformance suite；
5. 通过 <code>explain_dispatch</code> 显示选择原因。

如果需要：

- 给 Gaussian 增加空字段；
- 修改 dispatch hard-code；
- 修改 core enum；
- import 第三方模型到 runtime；

则扩展设计判定失败。

---

## 10. 逻辑模块到仓库的投影

这部分只用于让依赖边界落地，不把“目录树”误当架构。

~~~text
src/splatforge/
  core/                 # contracts, registry, policy, planning, errors
  camera/               # camera semantic operators
  spatial/              # representation-neutral spatial operators
  sampling/             # sampling/warp operators
  matching/             # NCC/ZNCC and matching semantics
  gaussian/             # Gaussian-specific views, ops, recipes
  splat/                # screen-space splat semantics
  mesh/                 # future mesh-specific ops
  providers/
    torch/               # readable references/decompositions
    triton/              # optimized provider
    native_cuda/         # bindings and native implementation
  integrations/         # optional third-party adapters

specs/                   # normative operator specs
provenance/              # per-op source and implementation lineage
tests/
  conformance/           # protected common tests
  properties/
  gradients/
  integration/
  architecture/
benchmarks/
  definitions/
  baselines/
  results/
docs/
  adr/
  rfc/
  operators/             # generated catalog
tools/
  schemas/
  quality/
~~~

一个新 operator 的正常 change series 应主要落在：

~~~text
spec + reference + one or more providers + tests + benchmark + provenance
~~~

但不能把它们塞进一个 PR。标准顺序是：

1. proposal/spec/reference/conformance/provenance；
2. human review 后冻结并创建 protected work order；
3. 独立 provider implementation PR；
4. 独立 verification/adversarial tests；
5. 可信 CI evidence 与 promotion PR。

Experimental 可以先合入 spec/reference，但不能在同一个 PR 连同实现直接晋级 candidate。

如果每次都要改 registry 核心分支、公共基类或其他 domain，就是架构摩擦信号。

---

## 11. 算子生命周期、支持矩阵与“完成”

不要把三类状态混成一个字段。

### 11.1 Operator maturity

| 状态 | 含义 |
|---|---|
| proposal | 只在设计阶段，不进入发行 API |
| experimental | 可试用，无兼容承诺 |
| candidate | 契约冻结，证据接近完整，邀请真实使用 |
| stable | 正式兼容承诺 |
| deprecated | 仍可使用，有替代和迁移窗口 |

出现静默错误或内存安全问题时可以将整个 operator 或某个 implementation quarantine；这不是正常成熟度倒退，而是安全状态。

### 11.2 Implementation cell conformance

先分清：

- declared eligibility：implementation manifest 的作者声明，只用于 runtime 候选过滤；
- verified support：可信 CI 对精确 artifact 和环境实际运行后产生；
- release tier：维护者批准的发布政策，implementation PR 无权修改。

按 <code>op × spec revision × implementation × artifact digest × device/GPU × dtype × grad mode × Python/PyTorch/CUDA stack</code> 记录 verified cell：

- <code>untested</code>；
- <code>unsupported</code>；
- <code>experimental</code>；
- <code>conformant</code>；
- <code>quarantined</code>。

### 11.3 Release tier

再单独标记：

- Tier 1 / release-required；
- Tier 2 / tested but non-blocking；
- best effort。

Tier 1 必须列出精确 Python、PyTorch、CUDA 和 GPU/compute-architecture 清单。范围式 declared eligibility 不能让未来 GPU 自动继承验证或 release tier。

因此“candidate operator 的某 artifact 在 CUDA/fp32/指定 GPU 与软件栈的 Tier-1 cell 已 conformant”是一个精确陈述；“SplatForge 支持 CUDA”不是。

### 11.4 晋级条件

Experimental 至少需要：

- spec 草案；
- provenance；
- 可读 reference；
- 基本 golden 与边界测试。

Candidate 还需要：

- 契约冻结；
- 至少一个优化 provider；
- differential/property/gradient tests；
- 每项 claimed capability 对应测试；未声明的 compile/export/fake/autograd 能力明确记录为 unsupported，并验证 dispatch 不会误选；
- GPU safety；
- benchmark；
- 真实支持矩阵。

Stable 还需要：

- 至少一个真实外部消费者；
- Tier 1 单元全部通过；
- 文档与迁移说明；
- provenance/license review；
- 维护 owner；
- release 级性能和安装证据。

“跨领域通用”只有在两个不同 representation/domain 实际使用后才能声称，不能靠抽象名字自封。

---

## 12. 正确性与 CI

### 12.1 正确性不能只有“CUDA 对 Torch”

reference 和优化实现可能共同误读公式。稳定算子至少需要四类独立证据：

1. **手工小例子**：人工可以推导；
2. **高精度 reference**：优先清晰的 CPU/float64；
3. **性质/变形测试**：平移、旋转、尺度、置换、分块等不变量；
4. **provider differential**：Torch/Triton/CUDA 互比。

支持梯度时再增加：

- gradcheck 或独立 finite difference；
- directional derivative；
- 不连续边界的安全间隔；
- 明确 nondifferentiable 输入；
- 若声称二阶梯度，则 gradgradcheck；
- 梯度累积、非 contiguous、多次 backward 和 mixed precision。
- non-contiguous/expanded upstream gradients，以及声明过的 backward copy 行为。

输入族至少覆盖：

- 空张量、单元素、非整块尺寸；
- 切片、非 contiguous 和不同 stride；
- 极大/极小值、NaN、Inf；
- 奇异 covariance、重复点、退化 ray/triangle；
- 多 CUDA stream、多 device；
- dynamic shape 和各 batch 形式；
- forward/backward；
- 同栈重复运行。

随机失败必须保存 seed、shape、dtype、provider、build fingerprint 和最小化复现输入。

容差按 <code>op × dtype × case class × precision profile</code> 维护。禁止全库一个宽松 atol。

### 12.2 PyTorch 组合能力

对承诺相应能力的 custom op，运行：

- <code>torch.library.opcheck</code>；
- fake/meta 完整 metadata inference（shape/dtype/device/layout/stride/storage offset/alias）；
- <code>torch.compile(fullgraph=True)</code> smoke；
- export smoke；
- autocast；
- forward/backward registration。

opcheck 只验证注册与 PyTorch 子系统组合，不证明公式正确，所以不能替代上述四类证据。

### 12.3 分层 CI

**每个公共 PR：5–10 分钟 CPU 快速门禁**

- format、lint、type；
- manifest/schema；
- public API snapshot；
- architecture import rules；
- Torch CPU reference；
- golden、小规模 property；
- 文档示例；
- SPDX/REUSE；
- dependency/license policy；
- work-order allowed-path；
- protected threshold/test 检查。

**可信变更：定向 GPU**

- build + clean install；
- Torch/Triton/CUDA differential；
- forward/backward；
- opcheck、compile、fake；
- empty/non-contiguous/stream/device；
- Compute Sanitizer memcheck；
- 短 benchmark canary。

**Nightly**

- 最低支持栈、推荐栈、最新支持栈的 pairwise matrix；
- 全量 random/property/gradient；
- wheel 安装后测试；
- 多 GPU 代际；
- 长一些的 sanitizer/fuzz。

**Weekly**

- memcheck → racecheck → initcheck → synccheck；
- 长时间 fuzz；
- 性能 A/B；
- peak memory、compile time、first-call；
- dispatcher/contract mutation tests。

**Release candidate**

- 从干净 tag 构建；
- 无源码 checkout、无编译器环境安装；
- Tier 1 全矩阵；
- sanitizer；
- public API/contract diff；
- wheel 内容和动态库审计；
- source/binary license bundle；
- SBOM、哈希和 provenance attestation；
- 原始 benchmark JSON 和环境指纹。

### 12.4 GPU CI 安全

公开 fork PR 绝不能自动在个人目标 GPU（当前为 RTX 4080 SUPER）、长期自托管 GPU 或带 secrets 的机器上执行。

安全流程：

1. 未可信 PR 只自动运行隔离的托管 CPU CI；
2. 维护者批准精确 head commit SHA；任何新 push 自动撤销旧批准；
3. GPU job 使用 default/base branch 中受保护的可信 workflow，把已审 commit 放入无 secret、无个人数据、无内网权限的一次性 GPU 环境；
4. 每次任务后销毁或重镜像 runner；
5. release credential 只进入受保护环境；
6. release credential 所在 runner 永不执行 PR 代码；
7. 禁止使用 <code>pull_request_target</code> checkout 并执行 fork 代码。

### 12.5 性能证据

分开报告：

- build/JIT time；
- first call；
- steady-state forward；
- steady-state backward；
- peak memory。

每个 benchmark 固定：

~~~text
benchmark_id
input generator + seed
real/synthetic workload
tiny/typical/stress/adversarial shape
dtype + precision profile
exact provider + implementation ID
data transfer included or excluded
warmup + sample count
synchronization method
GPU/driver/PyTorch/CUDA/Triton
latency distribution + throughput + peak memory
~~~

性能门槛按算子设置，不用一个全库阈值。每个 release-blocking benchmark 在实现任务开始前就必须有机器可读 threshold 和 noise rule。默认可采用“相对回归 + 校准噪声”双条件，例如超过 5% 且超过 3×MAD 才 hard fail。

基线锁定到正式 release 或明确 commit。实现 PR 不得同时修改 baseline。baseline 只能通过受保护的独立 PR 更新，并同时附旧/新原始结果。agent 自产 benchmark JSON 只有 self-reported 调试价值。性能豁免必须单独审批、有 issue 和到期版本；正确性不能豁免，只能修复、降级支持或 quarantine。

### 12.6 可信政策根与权威 evidence

所有门禁都必须建立在贡献者不能随 PR 一起改掉的可信执行根上：

- <code>main</code> 禁止 direct push 和 force push；
- required checks、human CODEOWNER approval 和“新 commit 撤销旧审批”开启；
- 使用 merge queue/<code>merge_group</code>，或强制分支在合并前与 base up to date；
- policy、CPU、GPU 和 digest checks 针对 prospective merge SHA，而不只针对可能过时的 PR head；
- specs、reference、conformance、tolerances、baselines、work orders、dependency policy、provenance、CODEOWNERS/rulesets、CI/release config、<code>tools/schemas/**</code>、<code>tools/quality/**</code>、evidence verifier/signer、promotion scripts 和 dependency resolver 都是 human-owned protected paths；
- 机器人不能充当要求中的 human approval；
- scope/policy checker 从 default/base branch 读取受保护规则，不能让 PR 用自己提交的 checker 判断自己；
- 检查 symlink、submodule、generated output、archive extraction 和路径规范化，禁止从 allowed path 穿透 protected path；
- release tag 不可移动，已发布版本不可原地替换。

trust-root 变更本身要求独立 human maintainer 审批，不能由 LLM implementation task 修改，也不能与使用该弱化规则的功能 PR 组成自动合并链。

构建供应链同时要求：

- Python/build/release dependencies 使用 lockfile 和 artifact hashes；
- GitHub Actions 固定到完整 commit SHA；
- workflow token 使用最小 permissions；
- 发布优先使用 OIDC/trusted publishing，不保存长期上传 token；
- 构建从已校验的 wheelhouse/install set 执行，不在 release job 临时解析漂移依赖；
- verifier、signer、resolver 与 builder 的版本/digest 一并进入 attestation。

如果合并方式生成了不同 commit/artifact，PR-head evidence 不能直接成为 main/release evidence；promotion 只能引用最终 merge artifact digest 的可信结果。

implementation agent 可以提交的运行结果只标记为 <code>self_reported</code>，用于调试，不进入权威 capability matrix。

权威 evidence 由可信 CI 直接生成、签名并写入受保护 artifact/evidence store。每份记录绑定：

~~~text
commit SHA
wheel/artifact digest
OpSpec digest
reference digest
conformance-suite digest
tolerance digest
benchmark-definition digest
environment/build fingerprint
exact implementation ID
~~~

任一 digest 改变，受影响的 conformance、performance 和 Tier 状态自动失效并重跑。不能用旧报告给新代码、新 wheel 或新规范背书。

---

## 13. 面向 LLM 的开发制度

用户不必亲自写大部分实现，但必须让机器无法“改答案同时改考卷”。

### 13.1 角色分离

| 角色 | 可以做 | 不能在同一变更做 |
|---|---|---|
| Architect / Spec owner | RFC、OpSpec、边界约定、支持目标 | 提交被本规范验收的优化实现 |
| Reference author | 高精度可读 reference、手工 fixtures | 修改优化 provider |
| Implementation agent | 指定 provider 实现、私有调试测试 | 改 spec、reference、容差、baseline、CI |
| Verification agent | 新增 adversarial/property tests、独立审计 | 改实现或放宽标准 |
| Human maintainer | stable、兼容破坏、许可例外、release | 不能仅凭 LLM 自述宣布完成 |

角色可以由不同 LLM 会话承担，但权限、上下文与 worktree 必须分开。

### 13.2 Work order

每个实现任务使用机器可读工单。工单由 maintainer 在实现分支创建前提交到 protected branch；实现 PR 只引用 task ID 和 digest，无权新建或修改自己的工单。

~~~yaml
task_id: SF-OP-0002-TRITON
base_commit: "<exact trusted commit>"
op_id: matching.patch_zncc@1
op_spec_revision: "sha256:<digest>"
goal: implement Triton provider for the frozen contract
network_mode: off              # off | allowlist_proxy
context_inventory_required: true
tool_access_log_required: true

non_goals:
  - change public API
  - add new algorithm variants
  - integrate a paper training repository

allowed_paths:
  - src/splatforge/providers/triton/matching/**
  - tests/provider_private/triton/matching/**
  - self_reports/SF-OP-0002-TRITON/**

forbidden_paths:
  - specs/**
  - src/splatforge/providers/torch/**
  - tests/conformance/**
  - tests/tolerances/**
  - benchmarks/baselines/**
  - .github/**
  - pyproject.toml

frozen_inputs:
  spec:
    path: specs/matching/patch_zncc/
    digest: "sha256:<digest>"
  reference:
    path: src/splatforge/providers/torch/matching/patch_zncc.py
    digest: "sha256:<digest>"
  conformance:
    path: tests/conformance/matching/patch_zncc/
    digest: "sha256:<digest>"
  tolerances:
    path: tests/tolerances/matching.patch_zncc.yaml
    digest: "sha256:<digest>"
  benchmark:
    path: benchmarks/definitions/matching.patch_zncc.yaml
    digest: "sha256:<digest>"

source_allowlist:
  - type: document
    url: "<exact paper or official documentation URL>"
    version_or_commit: "<exact version>"
    content_digest: "sha256:<archived-content-digest>"
  - type: repository_files
    repository: "<exact repository URL>"
    commit: "<exact commit>"
    allowed_blobs:
      - path: "<allowed/file/path>"
        blob_digest: "sha256:<digest>"
        spdx_expression: "Apache-2.0"
    unlisted_paths: deny
    submodules: deny_unless_separately_listed
source_denylist:
  - non-commercial reference implementations
  - unknown-license snippets

required_commands:
  - manifest validation
  - conformance tests with provider fixed to triton
  - gradient tests
  - compile/fake tests if declared
  - benchmark against frozen baseline

required_self_reports:
  - environment.self_reported.json
  - conformance.self_reported.json
  - gradients.self_reported.json
  - benchmark.self_reported.json
  - known_limitations.md

stop_conditions:
  - contract appears inconsistent
  - required source has incompatible license
  - correctness requires changing frozen files
~~~

scope checker 必须从 base/default branch 读取工单和 checker，不信任 PR 自带版本，并验证：

- PR head 是工单 base commit 的后代；
- task ID、work-order digest 和所有 frozen digests 匹配；
- diff 只触碰 allowed paths；
- allowed path 内没有 symlink、submodule、路径穿透或会覆盖 protected files 的生成器；
- source allowlist 是精确 URL/version/commit/path/blob hash/SPDX，不是“论文”“官方文档”或“整个宽松仓库”这种宽泛类别；未列文件默认拒绝，submodule 作为独立项目解析。

实现环境默认断网；批准资料预先镜像并校验 digest。确需联网时只能通过 allowlist proxy，并保存工具访问日志和实际上下文清单摘要。CI 不能仅依赖 agent 自报“没有看过”。

agent 的 self-reported 文件没有发布权威性；可信 CI 根据第 12.6 节重新运行并生成正式 evidence。

### 13.2.1 跨 PR 不能绕过冻结

“实现不能改考卷”是跨提交链的规则，不只是单个 diff 检查：

- work order 固定 spec/reference/conformance/tolerance/benchmark digests；
- 上述任一内容的变更需要独立 PR、独立 verifier 和 human CODEOWNER；
- 容差放宽必须附失败样本、数值依据和契约兼容性分类；
- baseline 更新使用独立受保护 PR，附旧/新原始结果，不能与产生回归的实现形成相邻自动合并链；
- 任一冻结 digest 变化后，所有受影响 implementation evidence 自动失效；
- 原 implementation 必须在新考卷上由可信 CI 重跑，不能继承旧绿色状态。

### 13.3 自动失败条件

- 同一实现 PR 修改 provider 与 spec/reference/tolerance；
- 删除、skip、xfail 或弱化失败测试；
- 修改 benchmark baseline；
- 新增未批准依赖；
- 触碰 release workflow、CODEOWNERS 或 secrets；
- 没有 provenance；
- 显式 provider 不支持却 silent fallback；
- 通过 silent cast/copy 改变行为；
- 只报告编译成功，没有数值、梯度、sanitizer、性能证据；
- 为非 GS 算子修改 Gaussian 类型；
- 复制受限源码或来源不明片段。

实现 agent 如果认为测试或规范错误，只能提交独立的 <code>CONTRACT_CHANGE_REQUEST</code>，由 spec owner 和 verifier 处理。

### 13.4 如何判断另一个 LLM 会不会写

第一次不要让它写完整 rasterizer。分两次验证：

**能力测试：<code>matching.patch_zncc@1</code> 的 Triton provider**

- 与当前需求直接相关；
- 数学定义清楚；
- 可做高精度 Torch oracle；
- forward/backward、边界与性能都可量化；
- 足以观察 indexing、mask、数值稳定性和融合能力。

**架构测试：<code>mesh.ray_triangle_intersect@1</code> 或 <code>spatial.ray_aabb_intersect@1</code>**

- 完全非 Gaussian；
- 必须在不修改 runtime/Gaussian 的前提下加入；
- 验证 representation、operator 和 provider 三轴是否真的解耦。

评分：

- 40% 数值、边界、梯度正确性；
- 20% 没有破坏依赖和抽象；
- 15% GPU safety；
- 15% 性能与显存；
- 10% 文档与 provenance。

另外记录架构摩擦：

- 修改多少 core 文件；
- 是否引入新公共概念；
- 是否复制 provider boilerplate；
- 是否需要认识其他 representation；
- 是否改变既有算子行为。

LLM 最终交付的是 patch、原始证据和已知限制；“完成”由独立 CI 判定。

### 13.5 关于 clean-room 的准确说法

新开 LLM 会话不能证明严格法律意义上的 clean-room，因为通用模型可能在预训练中见过公开仓库。除非模型提供者能对目标仓库的训练数据给出保证，否则对外应称：

> 有来源控制和隔离记录的独立重实现。

而不是绝对 clean-room。

推荐证据链：

1. spec 作者只使用论文、数学定义、允许来源和自有需求；
2. implementation agent 只获得冻结 spec、reference interface 和允许来源；
3. verifier 从不变量、手算 fixture 和 finite difference 独立补测试；
4. 人工审查异常相似的变量名、常数、控制流和注释；
5. 记录会话、work-order hash、base commit、来源暴露与贡献者声明。

版权来源干净也不等于专利风险自动消失。

---

## 14. 许可证、来源与供应链

### 14.1 许可决策

SplatForge 自有代码采用 Apache License 2.0，并使用 DCO 接收贡献。

原因：

- 允许商业和闭源集成；
- 包含明确的贡献者专利授权条款；
- DCO 在项目早期比 CLA 摩擦小；
- 与“技术影响力优先、核心公开”的目标一致。

但只能准确宣称：

> SplatForge 自有代码以 Apache-2.0 发布；每个发行物附带依赖与许可证清单。

不能宣称整条第三方 pipeline 一定没有版权、copyleft 或专利风险。

### 14.2 仓库级要求

- 顶层 LICENSE；
- 法律所需内容的 NOTICE；
- 每个源码/fixture 的 SPDX 标识；
- REUSE 检查；
- THIRD_PARTY.yml；
- source 与 binary wheel 分开的 license inventory；
- CITATION.cff；
- 每个算法的引用与 provenance；
- 发布 SBOM 和构建来源证明。

依赖按用途分类：

~~~text
runtime
build
test
benchmark
example
dataset/model
~~~

runtime、链接进二进制的 build library、vendored/bundled source 或 binary 默认只接受机器 allowlist 中批准的宽松许可证。non-commercial、unknown、禁止再分发的内容直接阻断 release；GPL/AGPL 虽不等于“禁止商业”，但其 copyleft 义务不符合 SplatForge 宽松核心目标，因此不得进入核心发行物。LGPL、MPL、专有 CUDA 组件需单独 RFC。

仅作为构建工具、既不链接也不分发的 tool-only dependency 单独登记和审查；不能把“出现在构建环境”笼统等同于进入发行物，也不能把上述项目策略冒充通用法律结论。

测试依赖和发布依赖严格分离，fixture、图片、模型权重和数据集也要记录许可证。

门禁使用机器可读的 SPDX allowlist/denylist 和带批准人、范围、到期条件的 exception record，并扫描：

- resolved transitive runtime dependencies；
- build/link dependencies；
- wheel 中的动态库与数据文件；
- submodules；
- vendored code；
- JIT、安装后或首次运行下载物；
- test/example/dataset/model artifacts。

unknown license 直接阻断发布。任何网络下载都必须预先在 manifest 中声明 URL、digest、license 和用途，不能在 import 或首次运行时偷偷拉取代码。

DCO 由承担责任的自然人签署；LLM 名称不能成为签署者。

### 14.3 每算子 provenance

~~~yaml
op_id: matching.patch_zncc@1
algorithm_sources:
  - type: paper
    title: ...
    version: ...
    url: ...
    accessed_at: 2026-09-12

allowed_implementation_sources:
  - official mathematical documentation
  - repository + exact commit + allowed paths/blob hashes + per-file SPDX

restricted_sources:
  - repository: ...
    reason: non-commercial or incompatible lineage

source_exposure:
  - actor: ...
    files_seen: ...
    date: ...

implementation:
  contributors: [...]
  work_order_hash: ...
  base_commit: ...
  third_party_code: []

test_fixture_sources: [...]
known_patent_status: unreviewed

review:
  provenance_reviewer: ...
  similarity_review: ...
~~~

受限源码不能交给 LLM 做“换变量名式重写”。论文公式可独立实现，但不自动清除第三方专利风险。

<code>source_exposure</code> 不只是记录字段，eligibility 覆盖 spec、reference、golden fixture、conformance test 和 implementation 的全部 normative actors。任何一环接触目标受限实现，都可能把受保护表达或实现 quirks 传进“考卷”。

如果相关 actor 接触过目标受限实现，对应产物不能直接标记为“独立重实现”证据。应换未接触该源码的人员/会话重做，或进入明确的人工法律/来源审查并降低对外声明。

受限程序的输出 snapshot 也不能直接成为 normative golden oracle，除非其生成、使用和再分发经过单独来源/法律审查。黑盒结果可以用于非规范性诊断，但正确性仍需来自数学、物理不变量和允许来源。

---

## 15. 版本、构建与发布

### 15.1 版本边界

包使用 SemVer，operator 另有 semantic major：

~~~text
camera.project_pinhole@1
matching.patch_zncc@1
gaussian.project_covariance@1
~~~

需要新 semantic major：

- 数学公式改变；
- 坐标/单位改变；
- tie-breaking 或排序改变；
- 输出 shape/含义改变；
- dtype promotion 改变；
- mutation/aliasing 改变；
- gradient 或 invalid-input 行为改变；
- 扩大已承诺的数值误差或削弱 precision profile 保证。

通常不需要新 major：

- 更快 kernel；
- 新 provider；
- 新 GPU arch；
- 私有 workspace 变化；
- 仍在已发布 tolerance 内的数值优化。

无版本 Python wrapper 绝不自动跟随“最新 semantic major”：

- <code>sf.camera.project_pinhole(...)</code> 在一个 package major 内固定绑定 <code>@1</code>；
- package minor 可以新增 <code>@2</code>，但只能通过显式 versioned API/OpKey 选择；
- 默认 wrapper 从 <code>@1</code> 切到 <code>@2</code> 只能随 package major；
- <code>@1</code> 在弃用窗口内继续存在；
- export/reproduction 记录实际 OpKey，而不是只有无版本函数名。

v0.x：

- experimental 可在 package minor 中快速变化，但每个构建的 spec revision 仍不可变且 implementation 必须精确匹配；
- candidate 表示该 semantic major 的可观察契约已冻结；澄清文字或加强测试会产生新 spec revision、使旧 evidence 失效并重跑；
- candidate 的任何可观察语义变化使用新 semantic major，不能靠 package minor 偷换；
- 有一组真正 stable 的算子后再发布 1.0。

1.x 以后：

- patch：实现修复，语义不变；
- minor：新算子/provider/dtype、兼容参数和 deprecation；
- major：不兼容契约变化。

stable 弃用窗口建议至少两个 minor release 且至少六个月，取更长者；删除只在 major。

### 15.2 明确不稳定的接口

v0.x 不承诺：

- <code>_C</code> symbols；
- <code>torch.ops.splatforge_native.*</code> 的外部直接调用；
- CUDA kernel ABI；
- C++ ABI；
- workspace 内存布局；
- serialized/export graph 的跨版本格式；
- provider 私有 autotune key。

遇到 silent-wrong-result 或 memory-safety 问题，可以在 patch release quarantine 某 implementation；公共 operator 保留，并发安全说明。

### 15.3 安装与二进制原则

冻结的是分发不变量，不提前冻结 wheel 拆分方式：

1. base install 不要求 nvcc；
2. native provider 缺失不导致顶层 import 失败；
3. Torch reference 是可运行路径，不只是测试文件；
4. 官方预编译 wheel 是成熟后的首选 UX；
5. native source/JIT build 是用户显式触发的长尾路径，不是默认安装体验；
6. import 时不编译全部 extension；
7. PEP 517 默认从 sdist 只构建 base/Torch 路径，不探测 GPU、不下载二进制、不调用 nvcc；
8. native build 只能由安装独立 provider 或安装后的显式命令触发；
9. 编译产物记录完整 Python/wheel、OS/libc、Torch ABI、CUDA/driver、compiler、SASS/PTX fingerprint；
10. native CUDA device code 与 ATen/PyTorch glue 尽量分离；
11. 是否使用当前 LibTorch stable ABI 必须先做真实 API coverage spike，不能只因文档存在就承诺；
12. CPython limited API 与 LibTorch stable ABI 是两个独立兼容轴。

v0.x 先保持一个源码仓库和一个 Python namespace。只有出现独立依赖、独立发布节奏或明显不同用户群后，再决定是否拆成 base/provider 多个发行包。

标准 wheel tag 不能表达完整 CUDA/GPU/Torch ABI 组合，不能依赖仍未正式落地的 wheel-variant 提案。首个 native 正式发行前必须通过 release RFC，在可实施方案中明确选择，例如独立 package index/local version 或独立 provider distribution，并写出 pip 的确定选择规则。

若拆 provider distribution，它使用唯一私有模块 namespace（例如 <code>splatforge_native_cuda</code>）并通过显式 entry point 注册；不得与 base wheel 重叠写入 <code>splatforge/</code> 文件。用户看到的稳定公共 namespace 仍只有 SplatForge API。

### 15.4 Release artifact

正式发布至少包含：

- sdist 与支持矩阵对应的 wheel；
- SHA-256；
- SBOM；
- source/binary license inventory；
- conformance report；
- capability matrix；
- 内置 build/capability manifest 与 detached signed release/advisory manifest；
- benchmark 原始 JSON；
- environment/build fingerprint；
- changelog；
- known limitations；
- provenance attestation。

发布采用 build-once/promote：

1. 可信 CI 只构建一次候选 sdist/wheel；
2. 安装、conformance、sanitizer、license、SBOM 和 benchmark 都针对同一 artifact digest；
3. release job 只提升这些已测试字节，禁止重新构建；
4. release manifest、evidence 和 provenance attestation 引用同一 digest；
5. 发布后的版本/tag/artifact 不可原地替换；有问题只能 yank/公告并发布新版本。

发布由 protected tag 和受保护工作流产生，不从开发者电脑直接上传。没有 bit-for-bit 证据时只说“可验证构建来源”，不声称 reproducible build。

### 15.5 Quarantine 与安全事件

quarantine 不能只改网站上的状态词。发现 silent-wrong-result、错误梯度或 memory-safety 问题时：

1. 冻结受影响 implementation 的自动选择；
2. 列出受影响 package version、artifact digest 和支持矩阵格子；
3. 发布 security/correctness advisory；
4. 必要时 yank 受影响发行物；
5. 立即发布禁用该实现或修复问题的 patch version；
6. 保留旧发行物和证据不可变，不回写历史；
7. 修复后重新生成全部权威 evidence，不能直接恢复旧状态。

patched release 必须携带更新后的 quarantine/advisory catalog。旧的离线安装无法被假装“远程撤销”，所以 advisory、yank 和升级通知仍是必要手段。

---

## 16. 实施路线

### Phase 0：先建立裁判系统

不写复杂 kernel，先完成：

- LICENSE、DCO、SPDX/REUSE 策略；
- ARCHITECTURE、首批 ADR；
- OpSpec 与 implementation manifest schema；
- registry、policy、structured error、explain_dispatch；
- architecture fitness tests；
- conformance harness；
- evidence/report schema；
- protected-file/work-order gate。

退出条件：一个虚拟 operator/provider 能完整注册、选择、解释、产出报告。

### Phase 1：最小纵切

实现 <code>camera.project_pinhole@1</code>：

- plain Torch reference；
- 明确 frame/pixel/batch/gradient contract；
- conformance、property、grad、compile tests；
- 一个简单优化 provider，主要用于验证 dispatch。

退出条件：第一次完整走通从 spec 到 release evidence 的生命周期。

### Phase 2：第一个真实有用算子

实现 <code>matching.patch_warp@1</code> 与 <code>matching.patch_zncc@1</code>：

- Torch 高精度 reference；
- Triton 作为优先优化候选；
- CUDA 只在 profiling 表明必要时加入；
- mask、boundary、zero variance、gradient 和 mixed precision 写进契约。

这也是测试另一个 LLM 编码能力的最佳起点。

### Phase 3：Gaussian geometry

按语义逐步加入：

- covariance parameterization；
- mean/covariance projection；
- screen bounds；
- ellipse coverage/compositing reference；
- 固定输出 schema 的 feature/geometry/statistics operators；
- native CUDA fused provider spike。

不要先暴露论文 CUDA 内部所有 micro-kernel。

### Phase 4：Rasterization 与真实消费者

- 少数 canonical output bundles；
- color/feature、alpha、expected depth；
- 再根据真实需求加入 median/visibility/per-element statistics；
- 外部 PGSR/Gaussian Wrapping 风格 adapter 作为消费者，不进入 core；
- 使用真实 workload 验证质量、速度和显存。

### Phase 5：证明不是 Gaussian 仓库

加入一个非 GS 垂直切片，例如 <code>mesh.ray_triangle_intersect@1</code> 或 <code>spatial.ray_aabb_intersect@1</code>。

硬退出条件：

- 不修改 runtime；
- 不修改 Gaussian；
- 不给旧 view 增加空字段；
- 公共 conformance/provider 注册机制原样复用。

---

## 17. 现在冻结与暂缓

### 17.1 现在冻结

- 项目定位：高性能空间算子系统，GS 是首个 domain；
- Apache-2.0 + DCO；
- PyTorch-first，Linux/NVIDIA 是首个优化 Tier 1；
- base install + Torch reference；
- 三条正交扩展轴；
- 无万能 Primitive/Scene/Renderer；
- <code>torch.Tensor</code> 是 v0.x frontend 数据边界，view 只在 API 边界，native kernel ABI 保持私有；
- OpSpec 与 ImplementationManifest 分离；
- provider 自动选择只限语义等价实现；
- 显式 provider 不 fallback；
- spec/reference/tests/baseline 与 implementation 权限隔离；
- per-cell support evidence；
- provenance 是发布门禁；
- v0.x 只承诺 Python semantic API；
- 初期单 repo、单 namespace。

### 17.2 暂缓

- 稳定 C++ ABI；
- LibTorch stable ABI 正式承诺；
- Windows/ROCm/MPS Tier 1；
- 多发行包拆分；
- 插件 marketplace；
- 通用场景图；
- 自定义 IR/compiler；
- 全局自动 tuning；
- LTS；
- 机器人专属对象模型；
- 完整训练或客户交付 pipeline。

暂缓不是拒绝，而是等待真实第二、第三用例提供压力。

---

## 18. 架构审查清单

任何新设计先回答：

### 抽象

- 它是新的数学语义、representation adapter，还是 provider？
- 是否错误地把三条轴混在一起？
- 是否接收了完整 model，而不是最小张量？
- 是否为了一个用例创建了全局概念？
- 第二个领域真的能复用，还是只有名字通用？

### 依赖

- dependency 是否只向薄腰流动？
- 新非 GS 功能是否触碰 Gaussian？
- provider 是否修改 public API？
- integration 是否被 core import？
- 没有 native extension 时能否 import？

### 语义

- frame、unit、pixel center、quaternion、batch 是否明确？
- empty/degenerate/NaN/tie 是否明确？
- dtype、accumulator、alias、mutation 是否明确？
- gradient 和不连续边界是否明确？
- dynamic output/capture 能力是否如实声明？

### 执行

- provider 是否只改变实现？
- dispatch 是否只读取 metadata？
- fallback、copy、cast 是否显式？
- current stream、device guard、workspace 生命周期是否正确？
- 是否可以解释选择结果？

### 证据

- 手算、reference、property、differential 是否彼此独立？
- gradient、sanitizer、compile 是否按声明运行？
- support matrix 是否只标记实际运行的格子？
- benchmark 是否固定 provider、同步方式和环境？
- 实现者是否改了考卷或 baseline？

### 法律与供应链

- 源代码、fixture、权重和依赖各自是什么许可证？
- provenance 是否记录到 commit/source？
- 是否接触受限源码？
- 是否把“独立重实现”错误宣传成严格 clean-room？
- 发布物是否包含 SBOM、license inventory、hash 和 build provenance？

---

## 19. 参考依据

以下只列与架构决策直接相关的官方一手资料，访问日期均为 2026-09-12。这里的 <code>main</code>、<code>latest</code> 与 GitHub branch URL 只用于设计期检索；真正的 implementation provenance 必须固定文档版本、release tag/commit 和可校验内容摘要。

- PyTorch custom operators 与推荐入口：  
  https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html
- PyTorch Python custom op、fake/meta 与 opcheck：  
  https://docs.pytorch.org/tutorials/advanced/python_custom_ops
- PyTorch custom op autograd registrations：  
  https://docs.pytorch.org/tutorials/advanced/python_custom_ops_registrations.html
- PyTorch C++/CUDA custom op 与 registration：  
  https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html  
  https://docs.pytorch.org/cppdocs/api/library/registration.html
- PyTorch 与用户定义 Triton kernel：  
  https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
- LibTorch stable ABI：  
  https://docs.pytorch.org/docs/main/notes/libtorch_stable_abi.html
- PyTorch C++/CUDA extension build 与 arch/PTX：  
  https://docs.pytorch.org/docs/stable/cpp_extension.html
- PyTorch reproducibility：  
  https://docs.pytorch.org/docs/stable/notes/randomness.html
- Python plugin discovery / entry points：  
  https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/  
  https://packaging.python.org/en/latest/specifications/entry-points/
- Wheel/packaging flow 与尚处 Draft 的 wheel variants：  
  https://packaging.python.org/en/latest/specifications/binary-distribution-format/  
  https://packaging.python.org/en/latest/flow/  
  https://peps.python.org/pep-0817/
- gsplat rasterization API 与数据约定：  
  https://docs.gsplat.studio/main/apis/rasterization.html  
  https://docs.gsplat.studio/main/conventions/data_conventions.html
- NVIDIA Warp 的 kernel、互操作、AOT 与资源模型：  
  https://nvidia.github.io/warp/latest/user_guide/programming_model.html  
  https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html  
  https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/reducing_compilation_and_startup_time.html
- PyTorch3D renderer 与 batching：  
  https://pytorch3d.org/docs/renderer  
  https://pytorch3d.org/docs/batching
- Kaolin representation、batching 与 conversion：  
  https://kaolin.readthedocs.io/en/latest/modules/kaolin.rep.surface_mesh.html  
  https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html  
  https://kaolin.readthedocs.io/en/latest/notes/conversions.html
- Open3D Tensor、point attributes 与 DLPack：  
  https://www.open3d.org/docs/latest/tutorial/core/tensor.html  
  https://open3d.org/docs/latest/tutorial/t_geometry/pointcloud.html
- nvdiffrast primitive 设计与许可证：  
  https://nvlabs.github.io/nvdiffrast/  
  https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt
- NVIDIA Compute Sanitizer：  
  https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
- NVIDIA CUDA binary/PTX 与 driver/runtime compatibility：  
  https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html  
  https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- GitHub 自托管 runner 安全：  
  https://docs.github.com/en/actions/reference/security/secure-use
- GitHub artifact attestations：  
  https://docs.github.com/en/actions/concepts/security/artifact-attestations
- Apache License 2.0 与应用方式：  
  https://www.apache.org/licenses/LICENSE-2.0.html  
  https://www.apache.org/legal/apply-license.html
- REUSE specification：  
  https://reuse.software/spec-3.3/
- Semantic Versioning：  
  https://semver.org/
- PGSR 官方许可：  
  https://github.com/zju3dv/PGSR/blob/main/LICENSE.md
- Gaussian Wrapping 官方许可与依赖入口：  
  https://github.com/diego1401/GaussianWrapping/blob/main/LICENSE.md  
  https://github.com/diego1401/GaussianWrapping/blob/main/install.py
- PaMO 官方仓库：  
  https://github.com/SarahWeiii/pamo

---

## 20. 最终判定

SplatForge 的核心抽象不应该是“Gaussian”，也不应该是“所有 3D 对象”。它应该是：

> 版本化数学契约 + 张量调用签名 + 可解释实现选择 + 独立证据。

Gaussian、point、mesh、3DV、CG、机器人代码可以在真正共享数学语义的地方共享 operator；不能共享的地方就保留 domain-specific operator。允许具体，拒绝耦合。

如果未来新增一个点云或 mesh 能力只需要增加它自己的 spec、view、provider、tests 和 provenance，而不修改 runtime 与 Gaussian，那么架构成功。反之，无论目录多漂亮，都是另一种论文代码堆。
