import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

LANG = "english"
STOP_WORDS = None
STEMMER = None

def load_nlp_resources():
    global STOP_WORDS, STEMMER
    if STOP_WORDS is None:
        STOP_WORDS = set(stopwords.words(LANG))
    if STEMMER is None:
        STEMMER = SnowballStemmer(LANG)

def read_text(filename):
    if not os.path.exists(filename):
        print(f"Fisierul {filename} nu exista!")
        return ""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, num_chunks):
    size = len(text)
    chunk_size = max(1, size // num_chunks)
    return [text[i:i + chunk_size] for i in range(0, size, chunk_size) if text[i:i + chunk_size].strip()]

def count_words(chunk):
    start = time.time()
    pid = os.getpid()

    load_nlp_resources()

    tokens = nltk.word_tokenize(chunk.lower())
    words = [w for w in tokens if w.isalpha() and w not in STOP_WORDS]
    stemmed = [STEMMER.stem(w) for w in words]

    elapsed = time.time() - start
    return Counter(stemmed), pid, len(stemmed), elapsed

def analyze_file(filename, num_processes):
    text = read_text(filename)
    if not text:
        return Counter(), 0, 0, 0

    chunks = split_text(text, num_processes)
    total_words = sum(len(chunk.split()) for chunk in chunks)
    total_count, total_processed = Counter(), 0
    start_total = time.time()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(count_words, chunk) for chunk in chunks]
        for f in as_completed(futures):
            result, pid, chunk_words, t = f.result()
            total_count.update(result)
            total_processed += chunk_words
            percent = (chunk_words / total_words) * 100 if total_words else 0
            print(f"PID={pid} -> {chunk_words} cuvinte ({percent:.2f}%) in {t:.4f}s")

    elapsed = time.time() - start_total
    print(f"\nAnaliza '{filename}' terminata in {elapsed:.2f}s | {len(total_count)} cuvinte unice\n")
    return total_count, total_processed, len(total_count), elapsed


def analyze_multiple_files(filenames, num_processes=os.cpu_count()):
    combined, total_words, total_time = Counter(), 0, 0.0
    for file in filenames:
        print(f"\nProcesam: {file}")
        count, words, _, elapsed = analyze_file(file, num_processes)
        combined.update(count)
        total_words += words
        total_time += elapsed
    return combined, total_words, len(combined), total_time

def plot_top_words(counts, top_n=10):
    if not counts:
        print("Nu exista date de afisat.")
        return
    words, freq = zip(*counts.most_common(top_n))
    plt.bar(words, freq, color="cornflowerblue")
    plt.title(f"Top {top_n} cuvinte semnificative (fara stopwords)")
    plt.xlabel("Cuvinte")
    plt.ylabel("Frecventa")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_performance(process_counts, times, base_time):
    speedups = [base_time / t if t > 0 else 0 for t in times]
    plt.figure(figsize=(8, 5))
    plt.plot(process_counts, times, marker="o", label="Timp total (s)")
    plt.plot(process_counts, speedups, marker="s", label="Speedup (x)")
    plt.title("Scalabilitate paralela - Timp si Speedup")
    plt.xlabel("Numar procese")
    plt.ylabel("Valoare")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filenames = [f.strip() for f in input("Fisiere (separate prin virgula): ").split(",") if f.strip()]
    if not filenames:
        print("Niciun fisier introdus.")
        raise SystemExit

    max_procs = os.cpu_count()
    process_counts = [1, 2, max_procs // 2 or 1, max_procs]
    times, base_time = [], None

    for n in process_counts:
        print(f"\nRulare cu {n} procese:")
        _, _, _, elapsed = analyze_multiple_files(filenames, num_processes=n)
        times.append(elapsed)
        if base_time is None:
            base_time = elapsed

    plot_performance(process_counts, times, base_time)

    final_count, _, _, _ = analyze_multiple_files(filenames, num_processes=max_procs // 2 or 1)
    print("\nTop 10 cuvinte din toate fisierele:")
    plot_top_words(final_count, top_n=10)
