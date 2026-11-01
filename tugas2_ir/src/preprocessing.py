"""Indonesian text normalization, stopword filtering, and stemming."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_STEMMER = None  # The project uses the fast, transparent rule-based stemmer below.


STOPWORDS = set("""
ada adalah adanya adapun agak akan akankah akhirnya aku amat anda andalah antar antara
antaranya apa apakah apalagi apabila arahnya artinya asal atas atau ataupun awal
bagai bagaikan bagaimana bagaimanakah bagi bagian bahkan bahwa beberapa begitu
begini beginian belakangan belakangnya belum belumlah benar berapa berapakah
berapapun berupa besar betul biasa biasanya bila bilamana bisa bisakah boleh
bolehkah bolehkah buat bukan bukankah bukanlah bukannya cara cukup cuma dahulu
dalam dan dapat dari daripada dekat demikian demikianlah dengan di dia diam
diantara dilakukan dirinya diri sendiri disini disitu dulu daripada dua
empat enggak entah guna gunakan hal hampir hanya hanyalah hari harus haruslah
harusnya hendak hendaknya hingga ia ialah ibu ikut ingat ingin ini inilah itu
itulah jadi jadilah jangan jangankan janganlah jauh jelas jika juga jumlah
justru kadang kadang-kadang kalau kami kamilah kamu kamulah kan kapan kapanpun
karena karenanya ke keadaan kepada kepadanya kedua keduanya keinginan kelihatan
kelihatannya kembali kemudian kemungkinan kemungkinanlah kenapa kepada kepadanya
keseluruhan ketika khususnya kini kinilah kira kira-kira kita kitalah kok kurang
lagi lagian lah lain lainnya lalu lama lamanya lanjut lanjutnya lebih lewat
luar macam maka makanya makin malah mampu mampu membuat memang memastikan memberi
memerlukan merupakan menjadi mungkin mungkinkah nah namun nanti nantinya nyaris
oleh olehnya pada padahal paling panjang pantas para pasti pastilah penting
per perlu perlunya pernah pula pun punya rasa saat saja sajalah saling sama
samalah sama-sama sampai sana sangat sangatlah satu saya sayalah se sebab
sebabnya sebagai sebagaimana sebagainya sebagian sebaik sebaiknya sebaliknya
sebanyak sebesar sebisanya sebuah sebut sebagai secara secukupnya sedang
sedangkan sedemikian sedikit sedikitnya segera sejak sejauh sejumlah sekadar
sekadarnya sekali sekaligus sekarang sekalian sekalipun sesuatu seluruh
seluruhnya semacam semakin sementara semisal semula sendiri sendirian serta
sering setidaknya setiap siapapun sini sini sinilah situ situasi suatu sudah
sudahkah sudahlah supaya tadi tadinya tak tanpa tapi tegas terhadap tentu
tentulah tertentu terdapat terakhir terasa terhadap tadi tiga tiba tiba-tiba
tidak tidakkah tidaklah tiga tiap tiap-tiap tujuh tujuan turut umumnya
ujar umumnya untuk usai waktu waktunya wah yaitu yakni yang
""".split())

# Preserve negation because it can affect Boolean/query meaning.
STOPWORDS.difference_update({"bukan", "tidak", "belum", "jangan", "tanpa"})
TOKEN_RE = re.compile(r"[a-z]+", re.IGNORECASE)
REPLACEMENT_CHAR = "\ufffd"


def normalize_text(text: object) -> str:
    """Normalize Unicode, repair replacement chars, case-fold and tokenize-ready text."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).replace(REPLACEMENT_CHAR, " ")
    value = value.replace("\u00a0", " ").casefold()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"[_\d]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=100_000)
def _stem(token: str) -> str:
    """Fast light Indonesian affix stemmer (conservative, not a full KBBI lemmatizer).

    Removes common inflectional suffixes, then one derivational prefix. The
    nasal prefixes restore the common initial consonant (menyanyi -> nyanyi,
    mengirim -> kirim, memukul -> pukul, menulis -> tulis).
    """
    original = token
    exceptions = {"menang", "menteri", "menengah", "menara", "mengapa", "berita", "berbagai", "berikut", "belajar", "terus", "terutama", "tersebut", "penyanyi", "menyanyi", "perdana"}
    if token in exceptions:
        return {"penyanyi": "nyanyi", "menyanyi": "nyanyi"}.get(token, token)
    if token in {"mengajak", "mengalami", "mengolah", "mengadakan"}:
        return {"mengajak": "ajak", "mengalami": "alami", "mengolah": "olah", "mengadakan": "ada"}[token]
    suffixes = ("kan", "annya", "annya", "nya", "lah", "kah", "tah", "pun", "an")
    for suffix in suffixes:
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break

    if len(token) > 6:
        # Nasalized meN-/peN- prefixes. Restore a deleted consonant only when
        # the remainder starts with a vowel; otherwise leave the root onset.
        for prefix, restore in (("meny", "s"), ("peny", "s"), ("meng", "k"), ("peng", "k"),
                                ("men", "t"), ("pen", "t"), ("mem", "p"), ("pem", "p")):
            if token.startswith(prefix):
                remainder = token[len(prefix):]
                restore_deleted = remainder[:1] in "aiueo"
                if prefix in {"meng", "peng"}:
                    restore_deleted = remainder in {"embang", "irim", "umpul", "ubah", "ukur"}
                token = (restore if restore_deleted else "") + remainder
                break
        else:
            for prefix in ("me", "pe", "ber", "bel", "ter", "di", "ke", "se"):
                if token.startswith(prefix) and len(token) > len(prefix) + 3:
                    token = token[len(prefix):]
                    break
    return token if len(token) >= 3 else original


def preprocess(text: object, remove_stopwords: bool = True, stem: bool = True) -> list[str]:
    normalized = normalize_text(text)
    tokens = TOKEN_RE.findall(normalized)
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    if stem:
        tokens = stem_tokens(tokens)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Stem terms individually using cached Indonesian affix rules."""
    return [_stem(token) for token in tokens]


def stemmer_name() -> str:
    return "custom light Indonesian affix stemmer (cached, reproducible)"
