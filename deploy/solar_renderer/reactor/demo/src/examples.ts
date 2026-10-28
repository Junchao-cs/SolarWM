import structureUrl from "@solarrenderer-example/structure.mp4?url&no-inline";
import referenceUrl from "@solarrenderer-example/first.png?url&no-inline";
import promptUrl from "@solarrenderer-example/prompt.txt?url&no-inline";

export const examples = [
  {id:"highway-reference", title:"Synthwave Highway", detail:"Official video + appearance + prompt", reference:true, thumbnail:referenceUrl},
  {id:"highway-structure", title:"Structure only", detail:"Same drive · no appearance image", reference:false, thumbnail:`${import.meta.env.BASE_URL}examples/highway/structure.png`},
] as const;

export async function loadExample(id:string) {
  const example=examples.find(item=>item.id===id);
  if(!example) throw new Error("Unknown example.");
  async function fetchAsset(url:string) {
    const response=await fetch(url);
    if(!response.ok) throw new Error(`Could not load example (HTTP ${response.status}). Please try again.`);
    return response;
  }
  const [video,image,text]=await Promise.all([
    fetchAsset(structureUrl).then(r=>r.blob()),
    example.reference ? fetchAsset(referenceUrl).then(r=>r.blob()) : Promise.resolve(null),
    fetchAsset(promptUrl).then(r=>r.text()),
  ]);
  // Match the tested reference-free mode; never leave an absent Video 2 in the prompt.
  const prompt=example.reference ? text : text.split("\n").filter(line=>!line.startsWith("<Video 2>")).join("\n")
    .replace("[keyframe completion + reference generation] The target video begins from <Video 2> and follows", "[reference generation] The target video follows")
    .replace("The shot begins from <Video 2>. ", "");
  return {
    structure:new File([video],"highway-structure.mp4",{type:"video/mp4"}),
    reference:image ? new File([image],"highway-first.png",{type:"image/png"}) : null,
    prompt, seed:0,
  };
}
