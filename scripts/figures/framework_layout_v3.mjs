// Method overview: semantic data objects, central action scoring, recurrent acquisition.
// Layout references are documented in figures/design_references/topconf_20260924/README.md.
export async function drawFramework({newFigure,source,shape,text,line,arrow,box,C,W}) {
  const f=newFigure('measurement_boundary_editable',W,336,
    'CBMJev: typed adaptive concept measurement. Original schematic, no empirical trajectory or probability. Public labels and cross-fitted automatic responses supply controller targets in the CUB protocol. Only the responder accesses raw x. The mask/action tokens illustrate legal sets, not a logged case. Design references: Post-hoc CBM https://openreview.net/forum?id=nA5AZ8CEyow ; BLIP-2 https://proceedings.mlr.press/v202/li23q.html ; active feature acquisition https://proceedings.mlr.press/v267/guney25a.html ; SWE-agent https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html . Reference images are not included in this slide.',[13,10.5,8.5]);
  const {s}=f;
  await source(f.r,'paper/cvpr2027/sections/method.tex');
  const pale={blue:'#EAF3FA',purple:'#F1ECF8',green:'#EAF6F0',gray:'#F4F5F7'};
  f.r.design={layout:'three grouped panels, evidence above candidate scoring, bottom acquisition return',
    panel_bounds:[[8,34,174,229],[199,34,307,229],[519,34,150,229]],pastel_palette:pale,
    references:['iclr2023-1303','icml2023-02','icml2025-0045','neurips2024-27']};
  function surface(x,y,w,h,fill,stroke='none',lw=0) {
    const r=shape(s,'roundRect',x,y,w,h,fill,stroke,lw,'panel');r.borderRadius=7;return r;
  }
  function t(str,x,y,w,h,pt=8.5,opts={}){return text(s,str,x,y,w,h,pt,opts);}
  function poly(pts,fill,stroke=C.blue,lw=1){
    const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys);
    const w=Math.max(...xs)-x,h=Math.max(...ys)-y;
    return s.shapes.add({geometry:'custom',name:'responder-model',position:{left:x,top:y,width:w,height:h},
      fill,line:{fill:stroke,width:lw},customPaths:[{width:w,height:h,commands:[...pts.map((p,i)=>({[i?'lineTo':'moveTo']:{x:p[0]-x,y:p[1]-y}})),{close:{}}]}]});
  }
  function tokens(x,y,active,color,size=7.5,gap=3){
    for(let i=0;i<5;i++)shape(s,'rect',x+i*(size+gap),y,size,size,active.includes(i)?color:C.white,
      active.includes(i)?color:C.gray,.6,'group-token');
  }
  function network(x,y,width,height,color){
    const cols=[[0,.5,1],[.15,.85],[.5]],nodes=cols.map((ys,i)=>ys.map(v=>[x+i*width/2,y+v*height]));
    for(let c=0;c<nodes.length-1;c++)for(const a of nodes[c])for(const b of nodes[c+1])line(s,[a,b],color,.6);
    for(const layer of nodes)for(const [a,b]of layer)shape(s,'ellipse',a-3,b-3,6,6,C.white,color,1,'model-node');
  }

  // The colored panels establish a hierarchy; data objects are separate from modules.
  t('(a) Measurement',10,4,176,25,13,{bold:true,color:C.blue});
  t('(b) Adaptive acquisition',201,4,307,25,13,{bold:true,color:C.purple});
  t('(c) Prediction',521,4,154,25,13,{bold:true,color:C.green});
  surface(8,34,174,229,pale.blue);
  surface(199,34,307,229,pale.purple);
  surface(519,34,150,229,pale.green);

  // A native input glyph: schematic image, not a fabricated dataset example.
  shape(s,'rect',69,54,51,37,C.white,C.blue,1,'input-image');
  shape(s,'ellipse',108,60,5,5,C.blue,C.blue,.5,'input-sun');
  line(s,[[72,86],[85,70],[95,81],[102,75],[117,86]],C.blue,1.2);
  t('Raw input x',33,97,125,18,10.5,{align:'center'});
  arrow(s,[[95,116],[95,132]],C.blue);
  poly([[35,133],[156,143],[156,180],[35,190]],C.blue);
  t('Responder',41,143,108,20,10.5,{bold:true,color:C.white,align:'center'});
  t('R(x, A)',41,167,108,16,8.5,{color:C.white,align:'center'});
  arrow(s,[[95,190],[95,210]],C.blue);
  t('Typed observations',23,232,147,19,8.5,{align:'center'});
  for(const [i,label]of ['ID','value','status'].entries()) {
    surface(24+i*48,211,43,21,C.white,C.blue,.8);t(label,24+i*48,212,43,18,8.5,{align:'center'});
  }
  arrow(s,[[165,221],[189,221],[189,89],[213,89]],C.green);

  // Acquired evidence is the shared input to scoring, candidate legality and prediction.
  surface(214,51,276,74,C.white,C.green,1.1);
  t('Acquired concept evidence',222,56,260,21,10.5,{bold:true,color:C.green,align:'center'});
  t('IDs · values · statuses',226,80,240,17,8.5,{align:'center'});
  t('Group mask',229,102,77,17,8.5);
  tokens(307,106,[0,2],C.green);
  t('observed',375,101,101,18,8.5,{color:C.green});
  // Empty history yields an instance-independent first action; no x path reaches here.
  arrow(s,[[274,125],[274,146]],C.green);
  arrow(s,[[421,125],[421,148]],C.green);

  t('Legal actions',216,143,116,19,10.5,{bold:true,color:C.purple});
  const actions=[['Singleton',[1]],['Pair',[3,4]],['All remaining',[1,3,4]],['STOP',[]]];
  for(const [i,[label,active]]of actions.entries()) {
    const y=167+i*20;
    t(label,214,y,80,16,8.5);
    tokens(289,y+4,active,C.purple,7,2);
  }
  t('unqueried groups',215,247,122,14,8.5,{color:C.gray});
  surface(350,149,139,66,C.white,C.purple,1.1);
  t('Action scoring',356,153,127,20,10.5,{bold:true,color:C.purple,align:'center'});
  t('Post-acquisition risk',354,178,131,17,8.5,{align:'center'});
  t('or signed loss reduction',351,195,138,16,8.5,{align:'center'});
  // A shared bracket routes the complete candidate family, including STOP.
  line(s,[[335,166],[340,166],[340,242],[335,242]],C.purple,.8);
  arrow(s,[[340,198],[349,198]],C.purple);
  // The declaration of remaining cost belongs to the decision, not the raw-input side.
  surface(350,220,139,37,C.purple);
  t('Acquire or STOP',354,221,131,19,10.5,{bold:true,color:C.white,align:'center'});
  t('declared budget',359,241,121,14,8.5,{color:C.white,align:'center'});
  arrow(s,[[420,215],[420,219]],C.purple);

  // Task-head internals are a generic model glyph; no claim about a new architecture.
  surface(537,63,116,86,C.white,C.green,1.1);
  t('Task head',543,66,104,20,10.5,{bold:true,color:C.green,align:'center'});
  network(558,96,71,22,C.green);
  t('fθ(H̄)',545,127,100,17,8.5,{align:'center'});
  arrow(s,[[490,89],[536,89]],C.green);
  arrow(s,[[595,149],[595,176]],C.green);
  shape(s,'diamond',588,177,14,14,C.white,C.green,1,'stop-gate');
  arrow(s,[[595,191],[595,212]],C.green);
  t('Prediction',539,214,113,20,10.5,{bold:true,align:'center'});
  t('ŷ',539,237,113,23,13,{bold:true,color:C.green,align:'center'});
  arrow(s,[[489,240],[514,240],[514,184],[587,184]],C.purple);
  t('STOP',533,162,52,18,10.5,{bold:true,color:C.purple});

  // The acquisition loop follows the perimeter, never crossing the main evidence path.
  arrow(s,[[420,257],[420,275],[18,275],[18,162],[34,162]],C.blue,1.5);
  surface(197,265,158,20,C.white);
  t('Acquire A; update evidence',201,265,151,19,8.5,{color:C.blue,align:'center'});

  // A compact training strip explains how the action-conditioned scorer is learned.
  surface(8,294,661,35,pale.gray);
  t('Training',16,300,63,18,10.5,{bold:true});
  t('Public labels + cross-fitted responses',85,296,238,16,8.5);
  arrow(s,[[319,305],[338,305]],C.gray,1,'dashed');
  t('Error / signed loss reduction',346,296,191,16,8.5);
  arrow(s,[[539,305],[558,305]],C.gray,1,'dashed');
  t('Fit controller',566,296,97,16,8.5);
  t('Declared cost controls acquisition; actual cost is logged externally.',85,313,568,15,8.5,{color:C.gray});
  return f;
}
