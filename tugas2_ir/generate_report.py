"""Generate an 8-page editable report from the executed evaluation artifacts."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Laporan_Tugas2_IR_Musik_Indonesia.docx"


def add_bullet(doc: Document, text: str) -> None:
    doc.add_paragraph(text, style="List Bullet")


def add_table(doc: Document, frame: pd.DataFrame, limit: int = 12) -> None:
    frame = frame.head(limit)
    table = doc.add_table(rows=1, cols=len(frame.columns))
    table.style = "Light Shading Accent 1"
    for i, col in enumerate(frame.columns):
        table.rows[0].cells[i].text = str(col)
    for row in frame.itertuples(index=False, name=None):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)


def main() -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(10)

    data = pd.read_csv(ROOT / "data" / "dataset_crawling_ir.csv", keep_default_na=False)
    pre = pd.read_csv(ROOT / "data" / "dataset_preprocessed.csv", keep_default_na=False)
    metrics = pd.read_csv(ROOT / "evaluation" / "metrics.csv")
    comparison = pd.read_csv(ROOT / "evaluation" / "ranking_comparison.csv")
    gt = pd.read_csv(ROOT / "evaluation" / "ground_truth.csv")
    means = metrics.groupby("model")[["precision@10", "recall@10", "f1@10"]].mean().round(4)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("TUGAS 2\nRETRIEVAL DATA DAN PEMBUATAN MESIN PENCARI SEDERHANA\n").bold = True
    p.add_run("Corpus Musik Indonesia — ANTARA News\nTemu Kembali Informasi")
    doc.add_paragraph("Nama kelompok: ........................................................\nAnggota: ........................................................\nKelas: ........................................................")

    doc.add_heading("BAB 1 — Pendahuluan dan Domain Dataset", level=1)
    doc.add_paragraph(f"Proyek ini melanjutkan Tugas 1 dengan mengolah {len(data)} artikel ANTARA News bertopik Musik Indonesia. Corpus berisi judul, URL, isi artikel, waktu crawling, dan metadata opsional. Dataset disaring pada Tugas 1 dengan aturan relevansi musik, kemudian pada Tugas 2 diproses menjadi koleksi dokumen yang dapat dicari.")
    doc.add_paragraph("Tujuan sistem adalah menerima query, memprosesnya secara konsisten dengan dokumen, menemukan kandidat melalui inverted index, memberi skor dengan TF-IDF + cosine atau BM25, dan menyajikan hasil Top-k beserta URL dan snippet.")
    add_bullet(doc, "Domain: antaranews.com; topik corpus: musik Indonesia, musisi, lagu, album, konser/festival, royalti, dan industri musik.")
    add_bullet(doc, "Input wajib: doc_id, title, url, content, crawl_date. Jumlah baris input: " + str(len(data)) + ".")
    add_bullet(doc, "Jumlah dokumen setelah preprocessing: " + str(len(pre)) + "; vocabulary index: " + str(len(json_read(ROOT / 'index' / 'inverted_index.json').get('postings', {}))) + " term.")
    doc.add_page_break()

    doc.add_heading("BAB 2 — Dasar Teori", level=1)
    doc.add_heading("2.1 Prapemrosesan dan Inverted Index", level=2)
    doc.add_paragraph("Prapemrosesan mengubah variasi permukaan teks menjadi token yang lebih konsisten. Inverted index memetakan term ke postings list dokumen yang mengandung term; setiap posting menyimpan doc_id dan TF. DF adalah banyaknya dokumen yang memuat term.")
    doc.add_heading("2.2 TF-IDF dan Cosine Similarity", level=2)
    doc.add_paragraph("TF log-normalized: tfw(t,d)=1+ln(tf(t,d)) untuk tf>0. IDF smooth: idf(t)=ln((N+1)/(df(t)+1))+1. Bobot TF-IDF adalah tfw×idf. Cosine similarity = (q·d)/(||q|| ||d||); skor lebih besar berarti arah vektor query dan dokumen lebih serupa.")
    doc.add_heading("2.3 BM25", level=2)
    doc.add_paragraph("BM25 menggunakan IDF(t)=ln(1+(N-df+0.5)/(df+0.5)) dan kontribusi term idf×tf(k1+1)/(tf+k1(1-b+b·|d|/avgdl)). Sistem menggunakan k1=1.2 dan b=0.75. Penyebut membuat pengaruh TF mengalami saturation; normalisasi panjang menyesuaikan dokumen panjang terhadap rata-rata.")
    doc.add_page_break()

    doc.add_heading("BAB 3 — Metodologi dan Implementasi", level=1)
    doc.add_heading("3.1 Prapemrosesan", level=2)
    doc.add_paragraph("Proses meliputi Unicode NFKC, case folding, pembuangan URL/tanda baca/angka, tokenisasi huruf, stopword removal, dan stemming imbuhan ringan Bahasa Indonesia. Stopword function words dibuang karena umumnya kurang membedakan topik; negasi seperti tidak/bukan/belum/tanpa dipertahankan karena mengubah makna. Stemming adalah aturan ringan deterministik untuk runtime; batasannya adalah cakupan morfologi tidak selengkap kamus Sastrawi. Dokumen kosong serta duplikasi URL/doc_id dibuang; teks pengganti Unicode pada input dinormalisasi.")
    if len(data):
        sample = data.iloc[0]
        processed = pre.loc[pre.doc_id.astype(str) == str(sample.doc_id), "preprocessed_text"]
        doc.add_paragraph("Contoh sebelum: " + (str(sample.title) + " — " + str(sample.content)[:350]))
        doc.add_paragraph("Contoh sesudah: " + (str(processed.iloc[0])[:450] if len(processed) else "(tidak ada)"))
    doc.add_heading("3.2 Index dan Retrieval", level=2)
    doc.add_paragraph("Index disimpan sebagai JSON dan dapat dibangun ulang dari dataset. TF disimpan per dokumen dan term, DF disimpan per term, serta panjang dokumen disimpan untuk BM25. Judul digabung dua kali dengan isi sebagai penguat judul yang transparan. Boolean Retrieval mendukung AND, OR, NOT, tanda kurung, dengan AND berprioritas lebih tinggi daripada OR.")
    doc.add_paragraph("Implementasi dibagi dalam preprocessing.py, indexing.py, boolean_retrieval.py, tfidf_retrieval.py, bm25.py, dan search_engine.py. app.py menyediakan CLI; streamlit_app.py menyediakan UI Streamlit.")
    doc.add_page_break()

    doc.add_heading("Contoh Index dan Perhitungan Bobot", level=1)
    index_payload = json_read(ROOT / "index" / "inverted_index.json")
    postings = index_payload["postings"]
    example_terms = [term for term in ("musik", "lagu", "album", "royalti") if term in postings]
    rows = []
    for term in example_terms:
        posting = postings[term][0]
        rows.append({"term": term, "DF": index_payload["df"][term], "contoh_doc_id": posting["doc_id"], "TF contoh": posting["tf"]})
    if rows:
        add_table(doc, pd.DataFrame(rows))
        doc.add_paragraph("Contoh numerik: untuk term t pada dokumen d dengan TF=f dan DF=df, TF log = 1+ln(f), IDF = ln((N+1)/(df+1))+1, dan bobot TF-IDF = TF log × IDF. Dengan N=" + str(index_payload["n_documents"]) + ", angka TF dan DF term contoh di atas dapat dimasukkan langsung ke rumus.")
        term = example_terms[0]
        posting = postings[term][0]
        doc.add_paragraph(f"Misal term '{term}', dokumen {posting['doc_id']}, TF={posting['tf']}, DF={index_payload['df'][term]}. TF log={1 + __import__('math').log(posting['tf']):.4f}; IDF={__import__('math').log((index_payload['n_documents'] + 1)/(index_payload['df'][term] + 1)) + 1:.4f}; TF-IDF adalah hasil kali keduanya.")
    doc.add_paragraph("Representasi query dan dokumen berupa dictionary vektor term → bobot. Cosine similarity membandingkan dot product kedua vektor setelah dibagi norm masing-masing. BM25 menggunakan postings yang sama, ditambah panjang dokumen serta rata-rata panjang dokumen.")
    from src.search_engine import SearchEngine
    from src.preprocessing import preprocess
    from collections import Counter
    from src.tfidf_retrieval import vectorize
    search_engine = SearchEngine(ROOT / "data" / "dataset_crawling_ir.csv")
    vector_query = "royalti lagu musik"
    query_vector = vectorize(Counter(preprocess(vector_query)), search_engine.tfidf.idf)
    top_doc = search_engine.search(vector_query, "tfidf", 1)[0]
    doc_vector = search_engine.tfidf.doc_vectors[top_doc["doc_id"]]
    doc.add_paragraph(f"Contoh VSM aktual — query '{vector_query}' → " + str({k: round(v, 4) for k, v in query_vector.items()}))
    shared = set(query_vector) & set(doc_vector)
    doc.add_paragraph(f"Dokumen {top_doc['doc_id']} '{top_doc['title']}' memiliki bobot term query bersama: " + str({k: round(doc_vector[k], 4) for k in sorted(shared)}))
    doc.add_page_break()

    doc.add_heading("BAB 4 — Hasil Eksperimen dan Evaluasi", level=1)
    doc.add_paragraph(f"Korpus input memuat {len(data)} dokumen; versi preprocessed memuat {len(pre)} dokumen. Evaluasi menjalankan {metrics.query_id.nunique()} query uji pada TF-IDF + cosine dan BM25. Ground truth rule-based menggunakan kriteria query/domain berbasis kecocokan istilah pada judul atau lead; file evaluasi menyimpan doc_id dan evidence agar label dapat diaudit.")
    doc.add_heading("4.1 Rata-rata metrik @10", level=2)
    add_table(doc, means.reset_index())
    doc.add_heading("4.2 Metrik per query", level=2)
    add_table(doc, metrics[["query_id", "model", "ground_truth_count", "precision@10", "recall@10", "f1@10"]], 20)
    doc.add_paragraph(f"Label ground truth: {len(gt)} query-document pasangan relevan. Hasil lengkap tersedia pada evaluation/metrics.csv dan evaluation/ground_truth.csv.")
    doc.add_page_break()

    doc.add_heading("Query Uji dan Ground Truth", level=1)
    query_specs = pd.read_csv(ROOT / "evaluation" / "queries.csv", keep_default_na=False)
    gt_examples = gt.groupby("query_id")["doc_id"].apply(lambda ids: ", ".join(map(str, ids.head(5)))).to_dict()
    gt_counts = gt.groupby("query_id").size().to_dict()
    query_summary = pd.DataFrame([{
        "ID": row.query_id,
        "Query": row.query,
        "Relevan": gt_counts.get(row.query_id, 0),
        "Contoh doc_id relevan": gt_examples.get(row.query_id, "—"),
    } for row in query_specs.itertuples()])
    add_table(doc, query_summary, 10)
    doc.add_paragraph("Ground truth dibentuk dengan kriteria reproducible per query: term yang tercantum di evaluation/queries.csv harus muncul sebagai kata/frasa utuh pada judul, atau sedikitnya dua term berbeda harus muncul pada 900 karakter awal isi. evaluation/ground_truth.csv menyimpan seluruh doc_id dan evidence; contoh doc_id ditampilkan di atas. Label kriteria ini merupakan baseline yang masih perlu spot-check manual.")
    doc.add_page_break()

    doc.add_heading("BAB 4 — Perbandingan Ranking", level=1)
    doc.add_paragraph("Overlap Top-10 menghitung dokumen bersama dua model. Nilai rendah menunjukkan perubahan urutan atau kandidat yang cukup besar; BM25 mempertimbangkan TF dan panjang dokumen melalui saturation/normalisasi, sementara cosine menormalisasi panjang vektor dan membobotkan term dengan IDF.")
    add_table(doc, comparison[["query_id", "query", "top10_overlap", "top10_jaccard"]], 10)
    max_diff = comparison.sort_values("top10_jaccard").head(1)
    if len(max_diff):
        doc.add_paragraph("Query dengan overlap Top-10 paling rendah: " + str(max_diff.iloc[0].query) + f" (Jaccard={max_diff.iloc[0].top10_jaccard:.3f}). Periksa ranking_comparison.csv untuk daftar Top-5 kedua model dan gunakan contoh konkret ini dalam pembahasan.")
    doc.add_heading("4.3 Contoh sistem", level=2)
    doc.add_paragraph("CLI: python app.py --query \"royalti lagu Indonesia\" --model bm25 --top-k 10. Antarmuka: streamlit run app.py. Output menampilkan peringkat, judul, URL, skor, dan snippet. Jika tidak ada hasil, sistem memberi pesan informatif.")
    doc.add_page_break()

    doc.add_heading("Analisis Query dengan Ranking Berbeda", level=1)
    doc.add_paragraph("Query dengan Jaccard Top-10 terendah dipilih sebagai contoh eksperimen. Periksa metrik dan daftar Top-5 kedua model untuk menghubungkan ranking dengan sifat model; contoh ini memperlihatkan bahwa kedua model memberi bobot berbeda pada frekuensi term, kelangkaan term, dan panjang dokumen.")
    if len(max_diff):
        qid = max_diff.iloc[0].query_id
        row = comparison.loc[comparison.query_id == qid].iloc[0]
        doc.add_paragraph(f"Query: {row['query']} (ID {qid}); Top-10 overlap={int(row['top10_overlap'])}, Jaccard={row['top10_jaccard']:.3f}.")
        doc.add_paragraph("Top-5 TF-IDF cosine (doc_id): " + str(row["tfidf_top5"]))
        doc.add_paragraph("Top-5 BM25 (doc_id): " + str(row["bm25_top5"]))
    doc.add_paragraph("Untuk query pendek/spesifik, term langka dapat memberi IDF tinggi. BM25 membatasi manfaat pengulangan term melalui TF saturation serta mengoreksi panjang dokumen, sedangkan cosine melakukan normalisasi vektor. Dua mekanisme ini menjelaskan perubahan urutan ketika dokumen panjang mengulang istilah query berkali-kali.")
    queries = pd.read_csv(ROOT / "evaluation" / "queries.csv", keep_default_na=False)
    short_ids = set(queries.loc[queries.experiment == "short_specific", "query_id"])
    short_comparison = comparison[comparison.query_id.isin(short_ids)][["query_id", "query", "top10_overlap", "top10_jaccard"]]
    doc.add_heading("Eksperimen query pendek dan spesifik", level=2)
    add_table(doc, short_comparison, 10)
    doc.add_heading("Keterbatasan Evaluasi", level=2)
    doc.add_paragraph("Ground truth saat ini dihasilkan dari kriteria yang terdokumentasi pada evaluation/queries.csv: kecocokan term domain pada judul atau lead. Ini reproducible dan dapat diperiksa, tetapi bukan label manusia. Sebelum melaporkan hasil sebagai evaluasi relevansi final, tinjau sampel positif dan negatif per query lalu koreksi evaluation/ground_truth.csv jika diperlukan.")
    doc.add_page_break()

    doc.add_heading("BAB 5 — Analisis, Kesimpulan, dan Pengembangan", level=1)
    doc.add_heading("5.1 Analisis", level=2)
    doc.add_paragraph("Precision mengukur proporsi hasil Top-10 yang relevan, recall proporsi ground truth yang ditemukan, dan F1 adalah harmonic mean keduanya. Nilai evaluasi bergantung pada ground truth berbasis aturan; label otomatis ini bukan anotasi ahli. Karena itu sampel hasil dan label perlu ditinjau manual, terutama query dengan banyak atau sedikit dokumen relevan.")
    doc.add_paragraph("TF-IDF cosine cenderung menonjolkan term informatif dan kemiripan distribusi vektor, sedangkan BM25 mengatur kontribusi term berulang secara non-linear dan memperhitungkan panjang dokumen. Perbedaan aktual perlu dikaitkan dengan query yang overlap-nya paling kecil sebagaimana tercatat di tabel eksperimen.")
    doc.add_heading("5.2 Kesimpulan dan pengembangan", level=2)
    doc.add_paragraph("Sistem telah mengimplementasikan prapemrosesan, inverted index, Boolean Retrieval, TF-IDF/VSM, cosine similarity, BM25, antarmuka pencarian, dan evaluasi sepuluh query. Pengembangan selanjutnya adalah mengganti label aturan dengan penilaian relevansi manusia, menguji Sastrawi pada korpus secara efisien, dan melakukan tuning parameter berbasis validasi.")
    doc.add_heading("Lampiran — Artefak", level=2)
    doc.add_paragraph("Dataset mentah: data/dataset_crawling_ir.csv. Dataset hasil preprocessing: data/dataset_preprocessed.csv. Index: index/inverted_index.json. Query: evaluation/queries.csv. Ground truth dan hasil evaluasi berada di folder evaluation/.")
    doc.save(OUT)
    print(f"Saved {OUT}")


def json_read(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
