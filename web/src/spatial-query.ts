/** Deterministic spatial questions over one loaded revision.
 *
 * Words map to catalog kinds, sides and intents of a closed vocabulary. Every
 * number is read from the revision's endpoint table and formatted with the
 * page's current model scale; nothing is estimated from the text, and a
 * missing fact is answered as missing. Open-vocabulary object search needs
 * the semantic text encoder and is not part of this resolver. */
export type QueryEndpoint = { id: string; objectId: string; label: string; side: "left" | "right" | null; measurementScope: string; heightNative: number };
export type QueryDifference = { id: string; label: string; minuendId: string; subtrahendId: string; valueNative: number };
export type QueryObject = { id: string; label: string; kind?: string };
export type QueryBend = { entityId: string; status: string; result?: { value: number } };
export type QueryInput<E extends QueryEndpoint = QueryEndpoint> = {
  objects: QueryObject[]; endpoints: E[]; differences: QueryDifference[]; bends?: QueryBend[];
  bound: (row: E) => boolean; format: (native: number) => string; revisionLabel: string;
};
export type QueryAnswer = {
  status: "answered" | "missing" | "unrecognized"; text: string; interpretation: string;
  objects: string[]; facts: { label: string; value: string; sourceId?: string }[];
};

const KINDS: { pattern: RegExp; kind: string; name: string }[] = [
  { pattern: /光幕|光栅|光电|安全光|light ?curtain/i, kind: "yellow safety post", name: "光幕" },
  { pattern: /围栏|护栏|栅栏|fence/i, kind: "safety fence", name: "围栏" },
  { pattern: /急停|按钮|emergency/i, kind: "emergency stop button", name: "急停按钮" },
  { pattern: /机器人|机械臂|robot/i, kind: "robot", name: "机械臂" },
  { pattern: /护板|guard/i, kind: "folded guard board", name: "折弯护板" },
  { pattern: /防撞柱|bollard/i, kind: "black bollard", name: "防撞柱" },
  { pattern: /小车|运输车|推车|cart/i, kind: "cart", name: "小车" },
  { pattern: /信号灯|状态灯|指示灯|signal|stack light/i, kind: "signal light", name: "信号灯" },
  { pattern: /标线|marking/i, kind: "floor marking", name: "地面标线" },
  { pattern: /标识牌|标牌|警示牌|sign\b/i, kind: "sign", name: "标识牌" },
  { pattern: /线缆托架|线槽|cable tray/i, kind: "cable tray", name: "线缆托架" },
];
const FLOOR = { pattern: /地面(?!标线)|floor/i, kind: "floor", name: "地面" };
const COMPARE = /谁更高|哪个更高|哪边更高|哪侧更高|谁高|哪个高|高差|差多少|相差|比较|对比|一样高|compare|higher|difference/i;
const BEND = /折弯|角度|夹角|angle|bend/i;
const CLEARANCE = /离地|多高|高度|底边|下沿|底端|clearance|height|high/i;
const sideName = { left: "左侧", right: "右侧" } as const;
const scopeName: Record<string, string> = { model_bottom_face_center: "模型底面中心", visible_face_lower_terminal: "可见面下沿", model_lower_rail_near_curtain: "光幕旁围栏下横梁底面" };

export function answerSpatialQuery<E extends QueryEndpoint>(question: string, input: QueryInput<E>): QueryAnswer {
  const text = question.trim();
  const kinds = KINDS.filter(row => row.pattern.test(text));
  if (!kinds.length && FLOOR.pattern.test(text.replace(/离地面?/g, ""))) kinds.push(FLOOR);
  const sides = (["left", "right"] as const).filter(side => (side === "left" ? /左/ : /右/).test(text));
  const intent = BEND.test(text) ? "bend" : COMPARE.test(text) ? "compare" : CLEARANCE.test(text) ? "clearance" : "locate";
  const interpretation = `对象：${kinds.map(row => row.name).join("、") || "未识别"} · 侧别：${sides.map(side => sideName[side]).join("、") || "未指定"} · 问题：${{ bend: "角度", compare: "比较高低", clearance: "离地高度", locate: "定位" }[intent]} · 版本：${input.revisionLabel}`;
  const result = (status: QueryAnswer["status"], answer: string, objects: string[], facts: QueryAnswer["facts"] = []): QueryAnswer => ({ status, text: answer, interpretation, objects: [...new Set(objects)], facts });
  if (!kinds.length) return result("unrecognized", `没有识别到对象类别。可查询：${[...KINDS, FLOOR].map(row => row.name).join("、")}。开放词汇搜索需要语义文本编码器，本页未运行。`, []);
  const ofKind = (kind: string) => input.objects.filter(object => object.kind === kind);
  const rowsOf = (kind: string, side?: "left" | "right") => input.endpoints.filter(row => ofKind(kind).some(object => object.id === row.objectId) && (!side || row.side === side));
  const fact = (row: E) => ({ label: `${row.label}离地`, value: input.bound(row) ? input.format(row.heightNative) : "未知（显示模型与测量模型不同）", sourceId: row.id });
  const endpoint = (id: string) => input.endpoints.find(point => point.id === id);
  // A difference is read only while both endpoints are on the displayed representations.
  const pairBound = (row: QueryDifference) => [row.minuendId, row.subtrahendId].every(id => { const point = endpoint(id); return !!point && input.bound(point); });
  if (intent === "bend") {
    const guards = ofKind("folded guard board").filter(object => !sides.length || sides.some(side => object.label.includes(sideName[side])));
    const measured = guards.map(object => ({ object, bend: input.bends?.find(row => row.entityId === object.id) })).filter(row => row.bend?.status === "measured" && row.bend.result);
    if (!measured.length) return result("missing", `本版本没有${kinds[0].name}的已测折弯角度。`, guards.map(object => object.id));
    return result("answered", measured.map(row => `${row.object.label}：${row.bend!.result!.value.toFixed(1)}°（模型估计，实物角度未唯一确定）`).join("；"), measured.map(row => row.object.id),
      measured.map(row => ({ label: `${row.object.label}折弯内角`, value: `${row.bend!.result!.value.toFixed(1)}°` })));
  }
  const kind = kinds[0];
  if (intent === "locate") {
    const objects = ofKind(kind.kind), sided = sides.length ? rowsOf(kind.kind).filter(row => row.side && sides.includes(row.side)).map(row => row.objectId) : [];
    const targets = sides.length ? sided : objects.map(object => object.id);
    return targets.length ? result("answered", `找到 ${targets.length} 个${kind.name}：${targets.map(id => input.objects.find(object => object.id === id)?.label ?? id).join("、")}。点击查看原图、模型和测量。`, targets)
      : result("missing", sides.length ? `本版本没有可判定左右的${kind.name}；左右只对有照片 4 测点的对象给出。` : `本版本没有${kind.name}对象。`, objects.map(object => object.id));
  }
  if (intent === "compare") {
    if (kinds.length >= 2 && kinds.some(row => row.kind === "yellow safety post") && kinds.some(row => row.kind === "safety fence")) {
      const pairs = input.differences.filter(row => row.id.endsWith("-minus-rail") && (!sides.length || sides.some(side => rowsOf("yellow safety post", side).some(point => point.id === row.minuendId))));
      if (!pairs.length) return result("missing", "本版本没有光幕与旁边围栏的成对测点。", []);
      const objectsOf = (rows: QueryDifference[]) => rows.flatMap(row => [row.minuendId, row.subtrahendId].map(id => endpoint(id)?.objectId ?? "")).filter(Boolean);
      const readable = pairs.filter(pairBound);
      if (!readable.length) return result("missing", "光幕与旁边围栏的测点没有绑定到当前显示模型，无法比较。", objectsOf(pairs));
      return result("answered", readable.map(row => `${row.label}：${input.format(row.valueNative)}（正值表示光幕测点更高）`).join("；")
        + (readable.length < pairs.length ? "；其余成对测点不在当前显示模型上，未比较。" : ""), objectsOf(readable),
        readable.map(row => ({ label: row.label, value: input.format(row.valueNative), sourceId: row.id })));
    }
    const left = rowsOf(kind.kind, "left"), right = rowsOf(kind.kind, "right");
    if (!left.length || !right.length) return result("missing", `本版本没有左右两侧的${kind.name}离地测点，无法比较。`, [...left, ...right].map(row => row.objectId));
    const difference = input.differences.find(row => left.some(point => point.id === row.minuendId) && right.some(point => point.id === row.subtrahendId));
    if (!difference) return result("missing", `本版本没有左右两个不同${kind.name}测点之间的高度差，无法比较。`, [...left, ...right].map(row => row.objectId));
    const a = left.find(point => point.id === difference.minuendId)!, b = right.find(point => point.id === difference.subtrahendId)!;
    if (!pairBound(difference)) return result("missing", `${kind.name}左右测点没有绑定到当前显示模型，无法比较。`, [a.objectId, b.objectId]);
    const value = difference.valueNative, higher = value > 0 ? "左侧" : value < 0 ? "右侧" : null;
    return result("answered", `${higher ? `${higher}${kind.name}测点更高` : "两侧测点一样高"}：左 − 右 = ${input.format(value)}。左 ${input.format(a.heightNative)}，右 ${input.format(b.heightNative)}（条件模型估计，未验证；不代表同高验证）。`,
      [a.objectId, b.objectId], [fact(a), fact(b), { label: difference.label, value: input.format(value), sourceId: difference.id }]);
  }
  const rows = sides.length ? sides.flatMap(side => rowsOf(kind.kind, side)) : rowsOf(kind.kind);
  if (!rows.length) {
    const objects = ofKind(kind.kind);
    return result("missing", `本版本没有${sides.map(side => sideName[side]).join("、")}${kind.name}的离地测量事实${objects.length ? "；可点击对象核对原图和模型，但不会估算离地值" : ""}。`, objects.map(object => object.id));
  }
  return result("answered", rows.map(row => `${row.label}离地 ${fact(row).value}（${scopeName[row.measurementScope] ?? row.measurementScope}；条件模型估计，未验证）`).join("；"), rows.map(row => row.objectId), rows.map(fact));
}
