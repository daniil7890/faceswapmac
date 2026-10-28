import Foundation

/// UI language. Stored in UserDefaults; default follows the system language.
enum Lang: String, CaseIterable, Identifiable {
    case ru, en
    var id: String { rawValue }
    var title: String { self == .ru ? "Русский" : "English" }

    private static let key = "uiLanguage"
    static var current: Lang {
        get {
            if let s = UserDefaults.standard.string(forKey: key), let l = Lang(rawValue: s) { return l }
            return Locale.preferredLanguages.first?.hasPrefix("ru") == true ? .ru : .en
        }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: key) }
    }
}

/// A string in both languages; resolved when shown, so switching language re-translates it.
struct Loc {
    let ru: String
    let en: String
    init(_ ru: String, _ en: String) { self.ru = ru; self.en = en }
    /// Same text for both languages (e.g. raw system error text).
    init(raw: String) { ru = raw; en = raw }
    var text: String { Lang.current == .ru ? ru : en }
}

/// Short helper: L("Привет", "Hello")
@inline(__always) func L(_ ru: String, _ en: String) -> String { Loc(ru, en).text }
