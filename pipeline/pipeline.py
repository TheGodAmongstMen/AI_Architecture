import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import shutil
from datetime import datetime, timezone
import json
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Any, Dict
import csv
from collections import Counter, defaultdict

# Pipeline utilities

def now_ms() -> int:
    return int(time.time() * 1000)


def log_event(stage: str, event: str, payload: Dict[str, Any]) -> None:
    rec = {"t_ms": now_ms(), "stage": stage, "event": event, "payload": payload}
    print(json.dumps(rec))

# AIS Schema

@dataclass(frozen=True)
class csv_ship:
    time_range: str
    flag: str
    vessel_name: str
    entry_time: str
    exit_time: str
    gear: str
    vessel_type: str
    mmsi: int
    imo: str
    callsign: str
    first_tx: str
    last_tx: str
    fishing_hours: float

@dataclass(frozen=True)
class valid_ship:
    time_range: str
    flag: str
    vessel_name: str
    entry_time: datetime
    exit_time: datetime
    gear: str
    vessel_type: str
    mmsi: int
    imo: str
    callsign: str
    first_tx: str
    last_tx: str
    fishing_hours: float


# Ingest input
def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def load_and_validate_csv(path: str) -> List[valid_ship]:
    start = now_ms()

    valid_rows = []
    rejected_rows = 0

    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                raw = csv_ship(
                    time_range=row["Time Range"],
                    flag=row["Flag"],
                    vessel_name=row["Vessel Name"],
                    entry_time=row["Entry Timestamp"],
                    exit_time=row["Exit Timestamp"],
                    gear=row["Gear Type"],
                    vessel_type=row["Vessel Type"],
                    mmsi=int(row["MMSI"]),
                    imo=row["IMO"],
                    callsign=row["CallSign"],
                    first_tx=row["First Transmission Date"],
                    last_tx=row["Last Transmission Date"],
                    fishing_hours=float(row["Apparent Fishing Hours"]),
                )
                
                validated = valid_ship(
                    time_range=raw.time_range,
                    flag=raw.flag,
                    vessel_name=raw.vessel_name,
                    entry_time=parse_datetime(raw.entry_time),
                    exit_time=parse_datetime(raw.exit_time),
                    gear=raw.gear,
                    vessel_type=raw.vessel_type,
                    mmsi=raw.mmsi,
                    imo=raw.imo,
                    callsign=raw.callsign,
                    first_tx=raw.first_tx,
                    last_tx=raw.last_tx,
                    fishing_hours=raw.fishing_hours,
                )

                if validated.fishing_hours < 0:
                    raise ValueError("negative fishing hours")

                if validated.entry_time >= validated.exit_time:
                    raise ValueError("entry time after exit time")

                valid_rows.append(validated)

            except Exception as e:
                rejected_rows += 1
                log_event("ingest", "row_rejected", {"error": str(e), "row": row})

    duration = now_ms() - start

    log_event("ingest", "summary", {
        "valid_rows": len(valid_rows),
        "rejected_rows": rejected_rows,
        "duration_ms": duration
    })


    return valid_rows


# Make documents more usable
def compute_dataset_stats(ships: List[valid_ship]):

    flags = Counter(s.flag for s in ships)
    gear_types = Counter(s.gear for s in ships)

    total_hours = sum(s.fishing_hours for s in ships)

    log_event("stats", "dataset_summary", {
        "ships": len(ships),
        "unique_flags": len(flags),
        "gear_types": dict(gear_types),
        "total_fishing_hours": total_hours
    })

def aggregate_vessels(ships: List[valid_ship]):

    vessels = defaultdict(list)

    for ship in ships:
        vessels[ship.mmsi].append(ship)

    summaries = []

    for mmsi, entries in vessels.items():

        vessel = entries[0]

        total_hours = sum(e.fishing_hours for e in entries)
        entry_count = len(entries)

        summaries.append({
            "mmsi": mmsi,
            "vessel_name": vessel.vessel_name,
            "flag": vessel.flag,
            "gear": vessel.gear,
            "entries": entry_count,
            "total_hours": total_hours,
            "first_seen": min(e.entry_time for e in entries),
            "last_seen": max(e.exit_time for e in entries)
        })

    log_event("stats", "vessel_summary", {
        "unique_vessels": len(summaries)
    })

    return summaries

def vessels_to_documents(vessels):

    docs = []

    for v in vessels:

        text = f"""
        Vessel {v['vessel_name']} flagged in {v['flag']} uses {v['gear']} fishing gear.

        MMSI: {v['mmsi']}.

        This vessel entered Brazil's EEZ {v['entries']} times.

        Total fishing activity recorded: {v['total_hours']} hours.
        """

        metadata = {
            "mmsi": v["mmsi"],
            "flag": v["flag"],
            "gear": v["gear"],
            "entries": v["entries"],
            "hours": v["total_hours"]
        }

        docs.append(Document(page_content=text.strip(), metadata=metadata))

    log_event("ingest", "documents_created", {"documents": len(docs)})

    return docs

def init_vectorstore():

    if os.path.exists(CHROMA_DB_PATH):
        print("Loading existing vectorstore...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embed_model
        )
        print(f"Loaded {vectorstore._collection.count()} chunks")
        return vectorstore

    print("Building new vectorstore...")

    ships = load_and_validate_csv(DATA_PATH)

    compute_dataset_stats(ships)    
    vessels = aggregate_vessels(ships)
    documents = vessels_to_documents(vessels)

    print(f"Converted {len(documents)} ships into documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=CHROMA_DB_PATH
    )

    print("Vectorstore saved")

    return vectorstore

# Configuration
DATA_PATH = "data.csv"
CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
# Init embed model
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# I only want errors, not warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Detect hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device
)


def query_llm(prompt, history=None):
    
    messages = [
        {"role": "system", "content": "You are a satellite above Brazil. Your job is to identify ships."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        temperature=0.5, # Low temp for more correct answers
        do_sample=True
    )
    
    # Isolate just the new tokens (the response)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def naive_rag(query):
    start = now_ms()

    # Retrieve with similarity scores
    results = vectorstore.similarity_search_with_score(query, k=3)

    retrieved_docs = []
    
    for i, (doc, score) in enumerate(results):
        retrieved_docs.append(doc)

    duration = now_ms() - start

    log_event("retrieval", "vector_search", {
        "query": query,
        "results": len(results),
        "latency_ms": duration
    })

    # Build context
    context_block = "\n\n".join([doc.page_content for doc in retrieved_docs])

    rag_prompt = f"""Use the context below to answer the question.

    Context:
    {context_block}

    Question: {query}
    """

    return query_llm(rag_prompt)

def hybrid_search(query, flag=None, gear=None):

    filters = {}

    if flag:
        filters["flag"] = flag

    if gear:
        filters["gear"] = gear

    results = vectorstore.similarity_search(
        query,
        k=5,
        filter=filters if filters else None
    )

    log_event("retrieval", "hybrid_search", {
        "query": query,
        "filters": filters,
        "results": len(results)
    })

    return results

def hybrid_rag(query, flag=None, gear=None):
    start = now_ms()

    flag, gear = extract_filters(query)

    results = hybrid_search(query, flag=flag, gear=gear)

    if not results:
        return "No vessels matched the query."

    context = "\n\n".join(doc.page_content for doc in results)

    rag_prompt = f"""
    Use the context below to answer the question.

    Context:
    {context}

    Question: {query}
    """

    duration = now_ms() - start
    log_event("retrieval", "hybrid_rag", {
        "query": query,
        "results": len(results)
    })
    raw_response = query_llm(rag_prompt)
    json_response = parse_llm_to_json(raw_response)
    return json_response

def extract_filters(query: str):

    flags = ["China","Spain","Brazil","Taiwan","Korea"]
    gears = ["trawler","longline","drift","purse seine"]

    detected_flag = None
    detected_gear = None

    for f in flags:
        if f.lower() in query.lower():
            detected_flag = f

    for g in gears:
        if g.lower() in query.lower():
            detected_gear = g

    return detected_flag, detected_gear

vectorstore = init_vectorstore()

# post-processing
llm_output_schema = {
    "vessel_name": str,
    "flag": str,
    "gear": str,
    "entries": int,
    "fishing_hours": float,
    "first_seen": str,   # ISO 8601
    "last_seen": str     # ISO 8601
}

def parse_llm_to_json(text: str) -> Dict:
    """
    Convert LLM free-text output about vessels into structured JSON.
    """

    def extract_field(pattern, default=None):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return default

    result = {
        "vessel_name": extract_field(r"Vessel\s+([A-Za-z0-9\s]+)", None),
        "flag": extract_field(r"flagged in\s+([A-Za-z]+)", None),
        "gear": extract_field(r"uses\s+([A-Za-z\s]+)\s+fishing gear", None),
        "entries": int(extract_field(r"entered.*?(\d+)\s+times", 0)),
        "fishing_hours": float(extract_field(r"Total fishing activity recorded:\s*([\d\.]+)", 0.0)),
        "first_seen": extract_field(r"first seen:\s*([\d\-\:TZ ]+)", None),
        "last_seen": extract_field(r"last seen:\s*([\d\-\:TZ ]+)", None)
    }

    return result

def main():
   
    print("\nSatellite:")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = hybrid_rag(user_input)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
