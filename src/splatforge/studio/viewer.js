"use strict";

const element = id => document.getElementById(id);

async function getJSON(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

class SceneView {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl");
    this.yaw = .35; this.pitch = -.15; this.zoom = .6;
    this.center = [0, 0, 0]; this.scale = 1; this.count = 0;
    this.triangles = false;
    if (!this.gl) throw new Error("WebGL is unavailable; 3D preview requires a WebGL-enabled browser.");
    this.initializeProgram();
    this.bindControls();
  }

  shader(type, lines) {
    const gl = this.gl, shader = gl.createShader(type);
    gl.shaderSource(shader, lines.join("\n")); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }

  initializeProgram() {
    const gl = this.gl;
    this.program = gl.createProgram();
    gl.attachShader(this.program, this.shader(gl.VERTEX_SHADER, [
      "attribute vec3 position; attribute vec3 color;",
      "uniform vec3 center; uniform float scale, yaw, pitch, zoom, aspect; varying vec3 c;",
      "void main() {",
      " vec3 p = (position-center)/scale; float a=cos(yaw), b=sin(yaw);",
      " p=vec3(a*p.x+b*p.z,p.y,-b*p.x+a*p.z); a=cos(pitch); b=sin(pitch);",
      " p=vec3(p.x,a*p.y-b*p.z,b*p.y+a*p.z);",
      " gl_Position=vec4(2.8*p.x/aspect,-2.8*p.y,.1*p.z,zoom+2.0+p.z);",
      " gl_PointSize=3.5; c=color; }"
    ]));
    gl.attachShader(this.program, this.shader(gl.FRAGMENT_SHADER, [
      "precision mediump float; uniform float point_mode; varying vec3 c;",
      "void main(){if(point_mode>.5 && distance(gl_PointCoord,vec2(.5))>.5)discard;",
      "gl_FragColor=vec4(c,1.0);}"
    ]));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
    gl.useProgram(this.program);
    this.buffers = {position: gl.createBuffer(), color: gl.createBuffer()};
  }

  bindControls() {
    let drag = null;
    this.canvas.onpointerdown = event => {
      drag=[event.clientX,event.clientY]; this.canvas.setPointerCapture(event.pointerId);
    };
    this.canvas.onpointerup = () => {drag=null;};
    this.canvas.onpointermove = event => {
      if (!drag) return;
      this.yaw += (event.clientX-drag[0])*.006;
      this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch+(event.clientY-drag[1])*.006));
      drag=[event.clientX,event.clientY]; this.draw();
    };
    this.canvas.addEventListener("wheel", event => {
      event.preventDefault(); this.zoom=Math.max(.2, Math.min(15,this.zoom+event.deltaY*.003)); this.draw();
    }, {passive:false});
    element("reset").onclick = () => {this.yaw=.35; this.pitch=-.15; this.zoom=.6; this.draw();};
    window.addEventListener("resize", () => this.draw());
  }

  update(data) {
    const gl=this.gl;
    this.count=data.points.length; this.triangles=!!data.triangles;
    if (this.count) {
      const min=[Infinity,Infinity,Infinity], max=[-Infinity,-Infinity,-Infinity];
      for (const point of data.points) for (let i=0;i<3;i++) {
        min[i]=Math.min(min[i],point[i]); max[i]=Math.max(max[i],point[i]);
      }
      this.center=min.map((value,i)=>(value+max[i])/2);
      this.scale=Math.max(...max.map((value,i)=>value-min[i]),.001);
    }
    for (const [name,values] of [["position",data.points],["color",data.colors]]) {
      gl.bindBuffer(gl.ARRAY_BUFFER,this.buffers[name]);
      gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(values.flat()),gl.STATIC_DRAW);
      const attribute=gl.getAttribLocation(this.program,name);
      gl.enableVertexAttribArray(attribute); gl.vertexAttribPointer(attribute,3,gl.FLOAT,false,0,0);
    }
    this.draw();
  }

  draw() {
    const gl=this.gl,canvas=this.canvas;
    canvas.width=canvas.clientWidth*devicePixelRatio; canvas.height=canvas.clientHeight*devicePixelRatio;
    gl.viewport(0,0,canvas.width,canvas.height);
    gl.clearColor(.063,.13,.21,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
    gl.uniform3fv(gl.getUniformLocation(this.program,"center"),this.center);
    const uniforms={scale:this.scale,yaw:this.yaw,pitch:this.pitch,zoom:this.zoom,aspect:canvas.width/canvas.height,point_mode:this.triangles?0:1};
    for (const [name,value] of Object.entries(uniforms)) gl.uniform1f(gl.getUniformLocation(this.program,name),value);
    gl.drawArrays(this.triangles?gl.TRIANGLES:gl.POINTS,0,this.count);
  }

  camera() {
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    const R=[[cy,0,sy],[sp*sy,cp,-sp*cy],[-cp*sy,sp,cp*cy]];
    const rotatedCenter=R.map(row=>row.reduce((sum,value,i)=>sum+value*this.center[i],0));
    const t=rotatedCenter.map((value,i)=>-value+(i===2?(this.zoom+2)*this.scale:0));
    const width=128,height=Math.max(1,Math.round(width*this.canvas.height/this.canvas.width));
    const focal=height*1.4;
    return {K:[[focal,0,width/2],[0,focal,height/2],[0,0,1]],world_to_camera:[...R.map((row,i)=>[...row,t[i]]),[0,0,0,1]],width,height,mode:element("mode").value};
  }
}

function drawHistory(history) {
  const canvas=element("chart"),context=canvas.getContext("2d");
  canvas.width=canvas.clientWidth*devicePixelRatio; canvas.height=canvas.clientHeight*devicePixelRatio;
  context.clearRect(0,0,canvas.width,canvas.height);
  if (history.length<2) return;
  const values=history.map(row=>row.loss),min=Math.min(...values),max=Math.max(...values);
  context.strokeStyle="#cf6535"; context.lineWidth=2*devicePixelRatio; context.beginPath();
  values.forEach((value,index)=>{
    const x=20*devicePixelRatio+index/(values.length-1)*(canvas.width-40*devicePixelRatio);
    const y=15*devicePixelRatio+(1-(value-min)/Math.max(max-min,1e-8))*(canvas.height-30*devicePixelRatio);
    if(index)context.lineTo(x,y);else context.moveTo(x,y);
  });
  context.stroke();
}

let scene,lastCheckpoint=-1,currentGeometry="cloud",polling=false;
try {scene=new SceneView(element("scene"));} catch(error) {element("error").textContent=error.message;}

async function updateGeometry() {
  const data=await getJSON("/"+currentGeometry);
  if(scene)scene.update(data);
  element("preview").textContent=data.points.length.toLocaleString();
  if(currentGeometry==="cloud")lastCheckpoint=data.checkpoint_step??-1;
}

async function poll() {
  if(polling)return;
  polling=true;
  try {
    const state=await getJSON("/state");
    element("stage").textContent=state.stage; element("step").textContent=(state.step||0).toLocaleString();
    element("method").textContent=state.method||"";
    element("count").textContent=state.gaussians===undefined?"—":state.gaussians.toLocaleString();
    if(state.loss!==undefined)element("loss").textContent=state.loss.toFixed(4);
    drawHistory(state.history||[]);
    if(state.export)element("export").textContent=state.export.exporter+"\n"+(state.export.vertices||0).toLocaleString()+" vertices · "+(state.export.faces||0).toLocaleString()+" faces";
    if(currentGeometry==="cloud"&&(state.step||0)>lastCheckpoint)await updateGeometry();
  } catch(error) {element("error").textContent=error.message;} finally {polling=false;}
}

element("geometry").onchange=async event=>{
  currentGeometry=event.target.value; element("error").textContent="";
  try{await updateGeometry();}catch(error){element("error").textContent=error.message;}
};
element("render").onclick=async()=>{
  const button=element("render"); button.disabled=true; element("error").textContent="";
  try {
    const url="/render?view="+encodeURIComponent(element("view").value)+"&mode="+element("mode").value;
    const response=await fetch(url);
    if(!response.ok)throw new Error((await response.json()).error||"Render failed");
    const blob=await response.blob(),previous=element("image").src;
    if(previous.startsWith("blob:"))URL.revokeObjectURL(previous);
    element("image").src=URL.createObjectURL(blob);
  } catch(error) {element("error").textContent=error.message;} finally {button.disabled=false;}
};
element("orbit-render").onclick=async()=>{
  if(!scene)return;
  const button=element("orbit-render");button.disabled=true;element("error").textContent="";
  try{
    const response=await fetch("/render-camera",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(scene.camera())});
    if(!response.ok)throw new Error((await response.json()).error||"Orbit render failed");
    const previous=element("image").src;if(previous.startsWith("blob:"))URL.revokeObjectURL(previous);
    element("image").src=URL.createObjectURL(await response.blob());
  }catch(error){element("error").textContent=error.message;}finally{button.disabled=false;}
};
poll(); setInterval(poll,5000);
