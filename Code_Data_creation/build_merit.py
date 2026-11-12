import json, ast, pickle, pandas as pd

df = pd.read_csv(r"C:\Document_UMONS\Thèse-Code\TD3_Minute\Data_Now\merit_order_dictionaries.csv", sep=";")
keys = pd.to_datetime(df["Datetime"]).astype("datetime64[ns]").astype("int64")

def parse(s):
    return json.loads(s)  # le format est du JSON valide (guillemets doubles, listes)

cache, retries = {}, []
for n, (k, a, b) in enumerate(zip(keys, df["incremental_merit_order"], df["decremental_merit_order"])):
    out = []
    for s in (a, b):
        for attempt in range(5):
            try:
                out.append(parse(s)); break
            except Exception as e:
                retries.append((n, attempt, type(e).__name__))
        else:
            raise RuntimeError(f"ligne {n} : échec après 5 essais")
    cache[int(k)] = tuple(out)

print("lignes :", len(cache), "| essais supplémentaires :", len(retries))
print(retries[:20])
with open(r"C:\Document_UMONS\Thèse-Code\TD3_Minute\Data_Now\merit_parsed.pkl", "wb") as f:
    pickle.dump(cache, f)
print ("done")