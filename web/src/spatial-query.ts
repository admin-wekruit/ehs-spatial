/** Deterministic spatial questions over one loaded revision.
 *
 * Words map to catalog kinds, sides and intents of a closed vocabulary. Every
 * number is read from the revision's endpoint table and formatted with the
 * page's current model scale; nothing is estimated from the text, and a
 * missing fact is answered as missing. Open-vocabulary object search needs
 * the semantic text encoder and is not part of this resolver. */
import { hasMessage, translate, type Language, type Params } from "./translate.ts";
export type QueryEndpoint = { id: string; objectId: string; label: string; side: "left" | "right" | null; measurementScope: string; heightNative: number; railPart?: string };
export type QueryDifference = { id: string; label: string; minuendId: string; subtrahendId: string; valueNative: number };
/** A comparison the revision refuses to make (the two points measure different parts), with its reason. */
export type QueryExclusion = { id: string; minuendId: string; subtrahendId: string; reason: string };
export type QueryObject = { id: string; label: string; kind?: string };
export type QueryBend = { entityId: string; status: string; result?: { value: number } };
export type QueryInput<E extends QueryEndpoint = QueryEndpoint> = {
  language: Language;
  objects: QueryObject[]; endpoints: E[]; differences: QueryDifference[]; excluded?: QueryExclusion[]; bends?: QueryBend[];
  bound: (row: E) => boolean; format: (native: number) => string; revisionLabel: string;
  /** The photo whose view defines left/right in this revision (the scene's reference photo). */
  sidePhoto?: number;
};
export type QueryAnswer = {
  status: "answered" | "missing" | "unrecognized"; text: string; interpretation: string;
  objects: string[]; facts: { label: string; value: string; sourceId?: string }[];
};

// ponytail: keep the original Chinese/English recognition rules in locale data; this does not broaden query semantics.
const inputPattern = (key: string, flags = "i") => new RegExp(translate("zh", "spatial.input." + key), flags);

const KINDS: { pattern: RegExp; kind: string; name: string }[] = [
  { pattern: inputPattern("curtain"), kind: "yellow safety post", name: "spatial.kind.curtain" },
  { pattern: inputPattern("fence"), kind: "safety fence", name: "spatial.kind.fence" },
  { pattern: inputPattern("emergency"), kind: "emergency stop button", name: "spatial.kind.emergency" },
  { pattern: inputPattern("robot"), kind: "robot", name: "spatial.kind.robot" },
  { pattern: inputPattern("guard"), kind: "folded guard board", name: "spatial.kind.guard" },
  { pattern: inputPattern("bollard"), kind: "black bollard", name: "spatial.kind.bollard" },
  { pattern: inputPattern("cart"), kind: "cart", name: "spatial.kind.cart" },
  { pattern: inputPattern("signal"), kind: "signal light", name: "spatial.kind.signal" },
  { pattern: inputPattern("marking"), kind: "floor marking", name: "spatial.kind.marking" },
  { pattern: inputPattern("sign"), kind: "sign", name: "spatial.kind.sign" },
  { pattern: inputPattern("cableTray"), kind: "cable tray", name: "spatial.kind.cableTray" },
  { pattern: inputPattern("gantry"), kind: "gantry", name: "spatial.kind.gantry" },
];
const FLOOR = { pattern: inputPattern("floor"), kind: "floor", name: "spatial.kind.floor" };
const COMPARE = inputPattern("compare");
const BEND = inputPattern("bend");
const CLEARANCE = inputPattern("clearance");
const sidePattern = { left: inputPattern("labelLeft", ""), right: inputPattern("labelRight", "") };

export function answerSpatialQuery<E extends QueryEndpoint>(question: string, input: QueryInput<E>): QueryAnswer {
  const t = (key: string, params?: Params) => translate(input.language, key, params);
  const list = (values: string[]) => values.join(t("spatial.listSeparator"));
  const sentences = (values: string[]) => values.join(t("spatial.sentenceSeparator"));
  const sideName = (side: "left" | "right") => t("spatial.side." + side);
  const scopeOf = (row: QueryEndpoint) => {
    const key = "spatial.scope." + (row.railPart === "lower_envelope_hypothesis" ? row.railPart : row.measurementScope);
    return hasMessage(key) ? t(key) : row.measurementScope;
  };
  const text = question.trim();
  const kinds = KINDS.filter(row => row.pattern.test(text));
  if (!kinds.length && FLOOR.pattern.test(text.replace(inputPattern("removeClearance", "g"), ""))) kinds.push(FLOOR);
  const sides = (["left", "right"] as const).filter(side => (side === "left" ? inputPattern("left", "") : inputPattern("right", "")).test(text));
  const intent = BEND.test(text) ? "bend" : COMPARE.test(text) ? "compare" : CLEARANCE.test(text) ? "clearance" : "locate";
  const interpretation = t("spatial.interpretation", { objects: list(kinds.map(row => t(row.name))) || t("spatial.unrecognized"), sides: list(sides.map(sideName)) || t("spatial.unspecified"), intent: t("spatial.intent." + intent), revision: input.revisionLabel });
  const result = (status: QueryAnswer["status"], answer: string, objects: string[], facts: QueryAnswer["facts"] = []): QueryAnswer => ({ status, text: answer, interpretation, objects: [...new Set(objects)], facts });
  if (!kinds.length) return result("unrecognized", t("spatial.noKind", { kinds: list([...KINDS, FLOOR].map(row => t(row.name))) }), []);
  const ofKind = (kind: string) => input.objects.filter(object => object.kind === kind);
  const rowsOf = (kind: string, side?: "left" | "right") => input.endpoints.filter(row => ofKind(kind).some(object => object.id === row.objectId) && (!side || row.side === side));
  const fact = (row: E) => ({ label: t("spatial.groundFact", { label: row.label }), value: input.bound(row) ? input.format(row.heightNative) : t("spatial.unboundValue"), sourceId: row.id });
  const endpoint = (id: string) => input.endpoints.find(point => point.id === id);
  // A difference is read only while both endpoints are on the displayed representations.
  const pairBound = (row: QueryDifference) => [row.minuendId, row.subtrahendId].every(id => { const point = endpoint(id); return !!point && input.bound(point); });
  if (intent === "bend") {
    const guards = ofKind("folded guard board").filter(object => !sides.length || sides.some(side => sidePattern[side].test(object.label)));
    const measured = guards.map(object => ({ object, bend: input.bends?.find(row => row.entityId === object.id) })).filter(row => row.bend?.status === "measured" && row.bend.result);
    if (!measured.length) return result("missing", t("spatial.noBend", { kind: t(kinds[0].name) }), guards.map(object => object.id));
    return result("answered", sentences(measured.map(row => t("spatial.bend", { label: row.object.label, angle: row.bend!.result!.value.toFixed(1) }))), measured.map(row => row.object.id),
      measured.map(row => ({ label: t("spatial.bendFact", { label: row.object.label }), value: `${row.bend!.result!.value.toFixed(1)}°` })));
  }
  const kind = kinds[0];
  if (intent === "locate") {
    const objects = ofKind(kind.kind), sided = sides.length ? rowsOf(kind.kind).filter(row => row.side && sides.includes(row.side)).map(row => row.objectId) : [];
    const targets = sides.length ? sided : objects.map(object => object.id);
    return targets.length ? result("answered", t("spatial.found", { count: targets.length, kind: t(kind.name), labels: list(targets.map(id => input.objects.find(object => object.id === id)?.label ?? id)) }), targets)
      : result("missing", sides.length ? t("spatial.noSidedObjects", { kind: t(kind.name), photo: input.sidePhoto ? t("spatial.photo", { number: input.sidePhoto }) : t("spatial.referencePhoto") }) : t("spatial.noObjects", { kind: t(kind.name) }), objects.map(object => object.id));
  }
  if (intent === "compare") {
    if (kinds.length >= 2 && kinds.some(row => row.kind === "yellow safety post") && kinds.some(row => row.kind === "safety fence")) {
      const pairs = input.differences.filter(row => row.id.endsWith("-minus-rail") && (!sides.length || sides.some(side => rowsOf("yellow safety post", side).some(point => point.id === row.minuendId))));
      const refused = (input.excluded ?? []).filter(row => row.id.endsWith("-minus-rail") && (!sides.length || sides.some(side => rowsOf("yellow safety post", side).some(point => point.id === row.minuendId))));
      if (!pairs.length) return result("missing", refused.length ? t("spatial.refused", { reasons: sentences(refused.map(row => row.reason)) }) : t("spatial.noRailPair"), []);
      const objectsOf = (rows: QueryDifference[]) => rows.flatMap(row => [row.minuendId, row.subtrahendId].map(id => endpoint(id)?.objectId ?? "")).filter(Boolean);
      const readable = pairs.filter(pairBound);
      if (!readable.length) return result("missing", t("spatial.unboundRailPair"), objectsOf(pairs));
      return result("answered", sentences(readable.map(row => t("spatial.railDifference", { label: row.label, value: input.format(row.valueNative) })))
        + (readable.length < pairs.length ? t("spatial.otherUnboundPairs") : "")
        + (refused.length ? t("spatial.otherSideRefused", { reasons: sentences(refused.map(row => row.reason)) }) : ""), objectsOf(readable),
        readable.map(row => ({ label: row.label, value: input.format(row.valueNative), sourceId: row.id })));
    }
    const left = rowsOf(kind.kind, "left"), right = rowsOf(kind.kind, "right");
    if (!left.length || !right.length) return result("missing", t("spatial.noSides", { kind: t(kind.name) }), [...left, ...right].map(row => row.objectId));
    const difference = input.differences.find(row => left.some(point => point.id === row.minuendId) && right.some(point => point.id === row.subtrahendId));
    const refused = (input.excluded ?? []).find(row => left.some(point => point.id === row.minuendId) && right.some(point => point.id === row.subtrahendId));
    if (!difference) return result("missing", refused ? t("spatial.refusedSides", { kind: t(kind.name), reason: refused.reason }) : t("spatial.noSideDifference", { kind: t(kind.name) }), [...left, ...right].map(row => row.objectId));
    const a = left.find(point => point.id === difference.minuendId)!, b = right.find(point => point.id === difference.subtrahendId)!;
    if (!pairBound(difference)) return result("missing", t("spatial.unboundSides", { kind: t(kind.name) }), [a.objectId, b.objectId]);
    const value = difference.valueNative, higher = value > 0 ? "left" : value < 0 ? "right" : null;
    return result("answered", t("spatial.sideDifference", { comparison: higher ? t("spatial.higher", { side: sideName(higher), kind: t(kind.name) }) : t("spatial.equalHeight"), value: input.format(value), left: input.format(a.heightNative), right: input.format(b.heightNative) }),
      [a.objectId, b.objectId], [fact(a), fact(b), { label: difference.label, value: input.format(value), sourceId: difference.id }]);
  }
  const rows = sides.length ? sides.flatMap(side => rowsOf(kind.kind, side)) : rowsOf(kind.kind);
  if (!rows.length) {
    const objects = ofKind(kind.kind);
    return result("missing", t("spatial.noClearance", { side: sides.length ? list(sides.map(sideName)) + (input.language === "zh" ? "" : " ") : "", kind: t(kind.name), action: objects.length ? t("spatial.checkObject") : "" }), objects.map(object => object.id));
  }
  return result("answered", sentences(rows.map(row => t("spatial.clearance", { label: row.label, value: fact(row).value, scope: scopeOf(row) }))), rows.map(row => row.objectId), rows.map(fact));
}
