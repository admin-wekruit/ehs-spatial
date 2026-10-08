// The message catalogs and the one lookup behind every UI string. Plain TypeScript (no JSX) so node tests and the viewer can import it.
import en from "./locales/en.json" with { type: "json" };
import zh from "./locales/zh.json" with { type: "json" };
import nl from "./locales/nl.json" with { type: "json" };

export type Language = "en" | "zh" | "nl";
export const LANGUAGES: Language[] = ["en", "zh", "nl"];
/** BCP 47 tag per language: the html lang attribute and the Intl locale of dates. */
export const LOCALE_TAG: Record<Language, string> = { en: "en-US", zh: "zh-CN", nl: "nl-NL" };
export type Params = Record<string, string | number>;
const catalogs: Record<Language, Record<string, string>> = { en, zh, nl };
export const isLanguage = (value: unknown): value is Language => LANGUAGES.includes(value as Language);
/** Message `id` in `language`, falling back to English and then to the id itself; `{name}` placeholders take `params`. */
export function translate(language: Language, id: string, params?: Params): string {
  const text = catalogs[language][id] ?? catalogs.en[id] ?? id;
  return params ? text.replace(/\{(\w+)\}/g, (match, name) => name in params ? String(params[name]) : match) : text;
}
/** Whether the catalogs know `id` (English is the complete catalog every other one falls back to). */
export const hasMessage = (id: string) => id in catalogs.en;
