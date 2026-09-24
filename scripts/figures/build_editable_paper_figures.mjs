#!/usr/bin/env node
// Native editable slide objects; all empirical marks are read from project data.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { drawFramework } from './framework_layout_v3.mjs';

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const runtime = process.env.RUNTIME_NODE_MODULES;
const skill = process.env.PRESENTATION_SKILL_DIR;
const python = process.env.RUNTIME_PYTHON;
if (!runtime || !skill || !python) throw new Error('Set RUNTIME_NODE_MODULES, PRESENTATION_SKILL_DIR, RUNTIME_PYTHON.');
const req = createRequire(path.join(runtime, '__figure_builder__.cjs'));
const { Presentation, PresentationFile } = await import(pathToFileURL(req.resolve('@oai/artifact-tool')).href);
const { finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const version = process.argv[2] ?? 'v1';
if (!/^v\d+$/.test(version)) throw new Error('Use a revision such as v1.');
const out = path.join(project, 'paper/cvpr2027/figures/editable', version);
const build = path.join(project, 'paper/cvpr2027/figures/editable/.build', version);
await fs.mkdir(out, { recursive: true });
await fs.mkdir(build, { recursive: true });
// The bundled headless renderer otherwise uses its private fallback fonts.
const fontDir=process.env.FIGURE_FONT_DIR ?? '/System/Library/Fonts/Supplemental';
await fs.access(path.join(fontDir,'Arial.ttf'));
await fs.mkdir(path.join(build,'font-cache'),{recursive:true});
await fs.writeFile(path.join(build,'fonts.conf'),`<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n<fontconfig><dir>${fontDir}</dir><cachedir>${path.join(build,'font-cache')}</cachedir></fontconfig>\n`);
const C = { ink:'#243247', blue:'#0072B2', orange:'#D55E00', green:'#009E73', purple:'#7A5DC7', gray:'#6E6E6E', white:'#FFFFFF' };
const W=676.8, H=336;
const presentation=Presentation.create({slideSize:{width:W,height:H}});
const figures=[];
const manifest={format:'cbmjev-editable-paper-figures-v1', version, font:'Arial', palette:C,
  author_approved_layout:true, editable_representation:'native PowerPoint text, shapes, paths and connectors',
  plot_editing:'Edit individual native marks, or regenerate from the accompanying source data; no Excel chart workbook.',
  canvas_px:[W,H], empirical_scope:'development validation; not locked-test or runtime evidence', figures};
let next=0;

function shape(s,geometry,x,y,w,h,fill='none',stroke='none',lw=0,name='shape') {
  return s.shapes.add({geometry,name:`${name}-${++next}`,position:{left:x,top:y,width:w,height:h},fill,
    line:{fill:stroke,width:lw,style:'solid'}});
}
function text(s,str,x,y,w,h,pt=8.5,{bold=false,color=C.ink,align='left'}={}) {
  const o=shape(s,'textbox',x,y,w,h,'none','none',0,'text');o.text=str;
  o.text.style={typeface:'Arial',fontSize:pt*4/3,bold,color,alignment:align,
    verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};
  return o;
}
function box(s,x,y,w,h,color=C.ink) {
  const b=shape(s,'roundRect',x,y,w,h,C.white,color,1.2,'module'); b.borderRadius=5;return b;
}
function line(s,pts,color=C.ink,width=1,style='solid',name='path') {
  const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const x=Math.min(...xs),y=Math.min(...ys),w=Math.max(.01,Math.max(...xs)-x),h=Math.max(.01,Math.max(...ys)-y);
  return s.shapes.add({geometry:'custom',name:`${name}-${++next}`,position:{left:x,top:y,width:w,height:h},
    fill:'none',line:{fill:color,width,style},customPaths:[{width:w,height:h,
      commands:pts.map((p,i)=>({[i?'lineTo':'moveTo']:{x:p[0]-x,y:p[1]-y}}))}]});
}
function arrow(s,pts,color=C.ink,width=1.3,style='solid') {
  line(s,pts,color,width,style,'flow'); const a=pts.at(-2),b=pts.at(-1);
  const t=Math.atan2(b[1]-a[1],b[0]-a[0]),len=5,wide=2.6;
  const p=[b,[b[0]-len*Math.cos(t)+wide*Math.sin(t),b[1]-len*Math.sin(t)-wide*Math.cos(t)],
    [b[0]-len*Math.cos(t)-wide*Math.sin(t),b[1]-len*Math.sin(t)+wide*Math.cos(t)]];
  const xs=p.map(v=>v[0]),ys=p.map(v=>v[1]),x=Math.min(...xs),y=Math.min(...ys);
  const w=Math.max(.01,Math.max(...xs)-x),h=Math.max(.01,Math.max(...ys)-y);
  s.shapes.add({geometry:'custom',name:`arrowhead-${++next}`,position:{left:x,top:y,width:w,height:h},fill:color,
    line:{fill:color,width:.2},customPaths:[{width:w,height:h,commands:[...p.map((v,i)=>({[i?'lineTo':'moveTo']:{x:v[0]-x,y:v[1]-y}})),{close:{}}]}]});
}
function marker(s,x,y,color,kind='ellipse',size=5,filled=true) {
  return shape(s,kind,x-size/2,y-size/2,size,size,filled?color:C.white,color,1.1,'data-marker');
}
function newFigure(id,w,h,notes,fonts) {
  const s=presentation.slides.add();s.background.fill=C.white;
  s.speakerNotes.textFrame.setText(notes);
  const crop={left:(W-w)/2,top:(H-h)/2,width:w,height:h};
  const record={id,slide:figures.length+1,crop_px:crop,font_sizes_pt:fonts,sources:[],points:[]};
  figures.push(record);return {s,r:record,ox:crop.left,oy:crop.top,w,h};
}
async function source(r,relative) {
  const b=await fs.readFile(path.join(project,relative));r.sources.push({path:relative,sha256:createHash('sha256').update(b).digest('hex')});return b.toString();
}
function legend(f,label,x,y,color,kind='ellipse',style='solid',filled=true) {
  line(f.s,[[f.ox+x,f.oy+y+7],[f.ox+x+18,f.oy+y+7]],color,1.2,style);
  marker(f.s,f.ox+x+9,f.oy+y+7,color,kind,4.5,filled);
  text(f.s,label,f.ox+x+24,f.oy+y,145,14,8);
}
function axes(f,{xmin,xmax,ymin,ymax,xticks,yticks,top=75,bottom=f.h-43,xlabel,ylabel}) {
  const {s,ox,oy,w}=f,left=35,right=w-11;
  const X=x=>ox+left+(x-xmin)/(xmax-xmin)*(right-left),Y=y=>oy+bottom-(y-ymin)/(ymax-ymin)*(bottom-top);
  text(s,ylabel,ox+5,oy+2,w-10,22,11,{bold:true});
  for(const y of yticks) {
    line(s,[[ox+left,Y(y)],[ox+right,Y(y)]],C.gray,.45,y===0?'solid':'dotted');
    text(s,String(y),ox+1,Y(y)-7,left-7,14,8,{align:'right'});
  }
  line(s,[[ox+left,oy+top],[ox+left,oy+bottom],[ox+right,oy+bottom]],C.ink,.85);
  for(const val of xticks) {
    const [x,label]=Array.isArray(val)?val:[val,String(val)];
    line(s,[[X(x),oy+bottom],[X(x),oy+bottom+3]],C.ink,.8);
    text(s,label,X(x)-14,oy+bottom+5,28,14,8,{align:'center'});
  }
  text(s,xlabel,ox+8,oy+f.h-21,w-16,18,9.5,{align:'center'});
  return {X,Y};
}
function curve(f,axis,points,color,kind='ellipse',style='solid',filled=true) {
  for(const p of points) if(p.sd) {
    const x=axis.X(p.x),lo=axis.Y(p.y-p.sd),hi=axis.Y(p.y+p.sd);
    line(f.s,[[x,lo],[x,hi]],color,.65);
    line(f.s,[[x-2.3,lo],[x+2.3,lo]],color,.65);
    line(f.s,[[x-2.3,hi],[x+2.3,hi]],color,.65);
  }
  if(points.length>1) line(f.s,points.map(p=>[axis.X(p.x),axis.Y(p.y)]),color,1.25,style,'data-curve');
  for(const p of points)marker(f.s,axis.X(p.x),axis.Y(p.y),color,kind,5,filled);
  f.r.points.push(...points);
}

// 1. Revised framework, informed by directly inspected top-conference references.
await drawFramework({newFigure,source,shape,text,line,arrow,box,C,W});

// 2. Initial single-seed diagnostic, preserving the original evidence scope.
{
  const f=newFigure('cub_budget_frontier_seed60_editable',315,254.4,
    'CUB validation, seed 60, 594 examples. Offline automatic-response replay. Source: results/main/cub_seed60_budget_grid/frontier.csv. Not test or latency evidence.',[11,9.5,8]);
  const csv=await source(f.r,'results/main/cub_seed60_budget_grid/frontier.csv');
  const rows=csv.trim().split(/\r?\n/),keys=rows.shift().split(',');
  const data=rows.map(row=>Object.fromEntries(row.split(',').map((v,i)=>[keys[i],i<2?v:Number(v)])));
  const a=axes(f,{xmin:-.8,xmax:29,ymin:0,ymax:40,xticks:[0,4,8,16,28],yticks:[0,10,20,30,40],top:72,bottom:207,xlabel:'Mean acquired concept groups',ylabel:'Validation accuracy (%)'});
  legend(f,'Fixed prefix',39,30,C.orange,'rect');legend(f,'Random prefix',187,30,C.gray,'diamond','dashed');
  legend(f,'All concepts',39,48,C.green,'diamond');legend(f,'No concepts',187,48,C.gray,'ellipse','solid',false);
  for(const [method,color,kind,style] of [['fixed',C.orange,'rect','solid'],['random',C.gray,'diamond','dashed'],['all',C.green,'diamond','solid']]) {
    const points=data.filter(r=>r.method===method).sort((u,v)=>u.mean_queried_groups-v.mean_queried_groups)
      .map(r=>({policy:r.policy_id,x:r.mean_queried_groups,y:r.accuracy*100}));curve(f,a,points,color,kind,style);
  }
  const stop=data.find(r=>r.method==='stop');curve(f,a,[{policy:stop.policy_id,x:0,y:100*stop.accuracy}],C.gray,'ellipse','solid',false);
}

// 3. Preserve all five matched policies, not just the easy baseline.
{
  const f=newFigure('cub_canonical_fixed_vs_dynamic_accuracy_editable',315,297.6,
    'CUB validation, seeds 60/61/62/63; the same 594 images recur. Error bars: sample SD across seeds. Offline replay, not runtime. Five original series preserved.',[11,9.5,8]);
  const report=JSON.parse(await source(f.r,'results/main/cub_canonical_fixed_replay_60_63/cub_canonical_fixed_replay_summary.json'));
  if(JSON.stringify(report.seeds)!=='[60,61,62,63]')throw new Error('Unexpected canonical seeds');
  const a=axes(f,{xmin:2,xmax:29,ymin:0,ymax:40,xticks:[4,8,16,28],yticks:[0,10,20,30,40],top:95,bottom:250,xlabel:'Mean acquired concept groups',ylabel:'Validation accuracy (%)'});
  const specs=[['value','Value',C.blue,'ellipse','solid',true,38,30],
    ['static','Fitted static',C.orange,'rect','solid',true,177,30],
    ['value_singleton','Value singleton',C.blue,'diamond','dashed',false,38,48],
    ['static_value','Fitted static + stop',C.orange,'triangle','dashed',false,177,48],
    ['canonical_fixed','Schema-order fixed',C.gray,'diamond','dotted',true,38,66]];
  for(const [policy,label,color,kind,style,filled,lx,ly] of specs) {
    legend(f,label,lx,ly,color,kind,style,filled);
    const points=report.budgets.map(k=>{const v=report.summary.policies[policy][String(k)];return {policy,budget:k,x:v.mean_queried_groups.mean,y:100*v.accuracy.mean,sd:100*v.accuracy.sd};});
    curve(f,a,points,color,kind,style,filled);
  }
}

// 4. Regenerate descriptive moments directly from per-seed source rows.
{
  const f=newFigure('cub_equal_mean_cost_gap_editable',315,254.4,
    'Retrospective equal-expected-declared-cost comparison against mixtures of adjacent fitted-static budgets. Repeated validation images. Error bars: sample SD across four seeds. X axis is maximum concept-group budget K, not measured latency.',[11,9.5,8]);
  const report=JSON.parse(await source(f.r,'results/main/cub_value_full_grid_equal_mean_cost_static_60_63_v1.json'));
  const budgets=[2,4,8,16,28];
  const a=axes(f,{xmin:-.18,xmax:4.18,ymin:-6,ymax:2,xticks:budgets.map((k,i)=>[i,String(k)]),yticks:[-6,-4,-2,0,2],top:67,bottom:207,xlabel:'Maximum concept groups (K)',ylabel:'Accuracy difference (pp)'});
  legend(f,'Value',39,31,C.blue);legend(f,'Value, singleton',162,31,C.blue,'rect','dashed',false);
  for(const [method,kind,style,filled]of [['value','ellipse','solid',true],['value_singleton','rect','dashed',false]]) {
    const points=budgets.map((k,i)=>{const p=report.policies[`${method}_K${k}`],v=p.rows.map(r=>100*r.adaptive_minus_static_mixture_accuracy);
      if(p.num_seeds!==4||p.rows.some(r=>r.split!=='validation'||r.num_samples!==594))throw new Error('Unexpected matched-cost rows');
      const mean=v.reduce((a,b)=>a+b,0)/v.length,sd=Math.sqrt(v.reduce((a,b)=>a+(b-mean)**2,0)/(v.length-1));
      if(Math.abs(mean-100*p.mean_accuracy_delta)>1e-9||Math.abs(sd-100*p.sample_std_accuracy_delta)>1e-9)throw new Error('Source moments disagree');
      return {policy:`${method}_K${k}`,budget:k,x:i,y:mean,sd,seed_deltas_pp:v};});
    curve(f,a,points,C.blue,kind,style,filled);
  }
}

// 5. Analytic pair boundary; no empty pseudo-empirical panel.
{
  const f=newFigure('analytic_boundary_editable',W,252,
    'Analytical illustration, not empirical performance. Independent symmetric flips and a Bayes head. v is a risk penalty, not milliseconds. Static pair purchase and pair-aware adaptive acquisition have equal Bayes performance in this construction. Source: sections/appendix.tex.',[13,10.5,8.5]);
  const {s,ox,oy}=f;await source(f.r,'paper/cvpr2027/sections/appendix.tex');
  text(s,'Analytical illustration',12,oy+2,640,23,13,{bold:true});
  text(s,'Risk penalty v',30,oy+36,260,19,10.5,{bold:true});
  const X=x=>ox+41+x/.5*267,Y=y=>oy+204-y/.5*132;
  for(const y of [0,.1,.2,.3,.4,.5]) {
    line(s,[[41,Y(y)],[308,Y(y)]],C.gray,.4,'dotted');text(s,y.toFixed(1),ox+3,Y(y)-7,30,15,8.5,{align:'right'});
  }
  line(s,[[41,oy+72],[41,oy+204],[308,oy+204]],C.ink,.85);
  for(const x of [0,.1,.2,.3,.4,.5])text(s,x.toFixed(1),X(x)-14,oy+208,28,15,8.5,{align:'center'});
  const pts=Array.from({length:101},(_,i)=>{const rho=i/200,v=(1-2*rho)**2/2;return [X(rho),Y(v)];});
  line(s,pts,C.blue,1.8);
  text(s,'Pair acquired',75,oy+172,130,17,8.5,{color:C.blue});
  text(s,'No pair',207,oy+102,97,17,8.5,{color:C.gray});
  text(s,'Noise ρ',95,oy+232,156,18,10.5,{align:'center'});
  line(s,[[336,oy+37],[336,oy+235]],C.gray,.6,'dashed');
  text(s,'v = (1 − 2ρ)² / 2',362,oy+45,300,26,13,{bold:true,color:C.blue});
  text(s,'Independent symmetric flips',362,oy+93,300,21,10.5);
  text(s,'Bayes task head',362,oy+124,300,21,10.5);
  text(s,'Static pair = pair-aware adaptive',362,oy+157,300,20,10.5);
  text(s,'Equal Bayes performance in this construction.',362,oy+183,303,17,8.5);
  text(s,'v is a risk penalty, not milliseconds.',362,oy+218,303,17,8.5,{color:C.gray});
  f.r.points=Array.from({length:101},(_,i)=>({rho:i/200,v:(1-i/100)**2/2}));
}

const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(presentation)).save(candidate);
await fs.writeFile(path.join(build,'figure_manifest.json'),JSON.stringify(manifest,null,2)+'\n');
const finalPath=path.join(out,'CBMJev_Figures.pptx');
await finalizePresentation({workspaceDir:project,candidatePath:candidate,finalPath,pythonExecutable:python,
  explicitTotalSlideCount:5,requiredNativeChartOwnerSlides:[],requiredNativeTableOwnerSlides:[],
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu',`${Math.round(W*9525)},${Math.round(H*9525)}`,'--validate-heading-fit'],
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
  receiptPath:path.join(build,'pptx_validation.json')});
await fs.copyFile(path.join(build,'figure_manifest.json'),path.join(out,'figure_manifest.json'));
console.log(JSON.stringify({pptx:finalPath,manifest:path.join(out,'figure_manifest.json'),slides:figures.length}));
