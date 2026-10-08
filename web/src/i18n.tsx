import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { isLanguage, LANGUAGES, LOCALE_TAG, translate, type Language, type Params } from "./translate";
export { LOCALE_TAG, translate, type Language, type Params } from "./translate";

const STORAGE_KEY = "panoptes.language";
/** ?lang= on the page, then the choice saved in this browser, then English. */
function initialLanguage(): Language {
  const query = new URL(location.href).searchParams.get("lang");
  if (isLanguage(query)) return query;
  try { const saved = localStorage.getItem(STORAGE_KEY); if (isLanguage(saved)) return saved; } catch { /* storage blocked: English */ }
  return "en";
}
type I18n = { language: Language; setLanguage: (next: Language) => void; t: (id: string, params?: Params) => string };
const Context = createContext<I18n>({ language: "en", setLanguage: () => {}, t: (id, params) => translate("en", id, params) });
export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setValue] = useState<Language>(initialLanguage);
  useEffect(() => { document.documentElement.lang = LOCALE_TAG[language]; }, [language]);
  const setLanguage = (next: Language) => {
    try { localStorage.setItem(STORAGE_KEY, next); } catch { /* storage blocked: the choice lasts for this page */ }
    setValue(next);
  };
  const t = (id: string, params?: Params) => translate(language, id, params);
  return <Context.Provider value={{ language, setLanguage, t }}>{children}</Context.Provider>;
}
export const useLanguage = () => useContext(Context);
export const useI18n = useLanguage;
/** The en / zh / nl switch; each option is named in its own language. */
export function LanguageSwitch() {
  const { language, setLanguage, t } = useLanguage();
  return <select aria-label={t("app.language")} value={language} onChange={(e) => setLanguage(e.target.value as Language)}>
    {LANGUAGES.map((code) => <option key={code} value={code}>{t("lang." + code)}</option>)}
  </select>;
}
