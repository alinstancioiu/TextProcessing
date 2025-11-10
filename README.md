# TextProcessing
Analiza și procesarea distribuită a textelor folosind tehnici de paralelizare

**Descriere proiect:**
Proiectul implementează un analizor de text care calculează frecvența fiecărui cuvânt din fișiere mari. Textul este împărțit în segmente, procesate simultan pe mai multe procese, iar rezultatele sunt combinate pentru a obține frecvența totală a tuturor cuvintelor. Rezultatele sunt afișate în consolă și pot fi vizualizate sub formă de grafic.

**Scop:**
Demonstrarea avantajelor paralelizării în procesarea de date mari;
Creșterea performanței și reducerea timpului de procesare față de analiza secvențială;
Obținerea unei analize complete a textului, cu frecvența tuturor cuvintelor, și prezentarea clară a rezultatelor.

**Tehnologii folosite:**
Python;
Paralelizare: concurrent.futures.ProcessPoolExecutor (procese);
Vizualizare: matplotlib.
