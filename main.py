from pyexpat import model
from fastapi import FastAPI, HTTPException,Query
import os
import subprocess
import json
import sqlite3
from datetime import datetime
import openai
from dotenv import load_dotenv
import re
import glob
import numpy as np
import shutil
import requests
from PIL import Image
import speech_recognition as sr
import git
from bs4 import BeautifulSoup
import pandas as pd
import markdown
from typing import Optional


load_dotenv()

app = FastAPI()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.get("/")
def read_root():
    return {"message": "LLM Automation Agent is running"}

@app.post("/run")
def run_task(task: str):
    """
    Process the given task and execute the appropriate function.
    """
    action = determine_task(task)
    
    if action:
        try:
            result = action()
            return {"status": "Success", "output": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Task could not be identified.")

@app.get("/read")
def read_file(path: str):
    """
    Read the contents of a file.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, "r") as f:
        content = f.read()
    return {"content": content}

def determine_task(task_description: str):
    """
    Use an LLM to determine which task the user wants to perform.
    """
    prompt = f"Identify the exact task among these:\n\n{task_description}\n\nTask Options:\n" \
             "A1: Install uv and run datagen.py\n" \
             "A2: Format /data/format.md using Prettier\n" \
             "A3: Count Wednesdays in /data/dates.txt\n" \
             "A4: Sort contacts in /data/contacts.json\n" \
             "A5: Get first lines of recent log files\n" \
             "A6: Create an index of markdown files\n" \
             "A7: Extract email sender from /data/email.txt\n" \
             "A8: Extract credit card number from /data/credit-card.png\n" \
             "A9: Find most similar comments in /data/comments.txt\n" \
             "A10: Calculate sales for Gold tickets in /data/ticket-sales.db\n\n" \
             "Return only the task ID (e.g., A1, A2, A3)."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    task_id = response["choices"][0]["message"]["content"].strip()

    task_mapping = {
        "A1": task_A1,
        "A2": task_A2,
        "A3": task_A3,
        "A4": task_A4,
        "A5": task_A5,
        "A6": task_A6,
        "A7": task_A7,
        "A8": task_A8,
        "A9": task_A9,
        "A10": task_A10,
    }

    return task_mapping.get(task_id)

# ---------------------------------------------------
# IMPLEMENTATION OF INDIVIDUAL TASKS
# ---------------------------------------------------

def task_A1():
    """
    Install `uv` (if required) and run `datagen.py` with the user’s email.
    """
    os.system("pip install uv")
    user_email = os.getenv("USER_EMAIL")
    os.system(f"python datagen.py {user_email}")
    return "Data generation completed."

def task_A2():
    """
    Format `/data/format.md` using Prettier.
    """
    os.system("npx prettier@3.4.2 --write /data/format.md")
    return "Formatted /data/format.md."

def task_A3():
    """
    Count the number of Wednesdays in `/data/dates.txt` and write to `/data/dates-wednesdays.txt`.
    """
    with open("/data/dates.txt", "r") as file:
        dates = file.readlines()
    
    count = sum(1 for date in dates if datetime.strptime(date.strip(), "%Y-%m-%d").weekday() == 2)
    
    with open("/data/dates-wednesdays.txt", "w") as file:
        file.write(str(count))
    
    return f"Counted {count} Wednesdays."

def task_A4():
    """
    Sort `/data/contacts.json` by `last_name`, then `first_name`.
    """
    with open("/data/contacts.json", "r") as file:
        contacts = json.load(file)
    
    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
    
    with open("/data/contacts-sorted.json", "w") as file:
        json.dump(sorted_contacts, file, indent=4)
    
    return "Sorted contacts.json."

def task_A5():
    """
    Write first line of the 10 most recent `.log` files in `/data/logs/` to `/data/logs-recent.txt`.
    """
    log_files = sorted(glob.glob("/data/logs/*.log"), key=os.path.getmtime, reverse=True)[:10]

    with open("/data/logs-recent.txt", "w") as output_file:
        for log_file in log_files:
            with open(log_file, "r") as file:
                first_line = file.readline().strip()
                output_file.write(first_line + "\n")

    return "Extracted first lines of recent logs."

def task_A6():
    """
    Create `/data/docs/index.json` mapping Markdown filenames to their H1 titles.
    """
    index = {}
    for md_file in glob.glob("/data/docs/*.md"):
        with open(md_file, "r") as file:
            for line in file:
                if line.startswith("# "):
                    index[os.path.basename(md_file)] = line[2:].strip()
                    break
    
    with open("/data/docs/index.json", "w") as file:
        json.dump(index, file, indent=4)

    return "Created Markdown index."

def task_A7():
    """
    Extract sender's email from `/data/email.txt`.
    """
    with open("/data/email.txt", "r") as file:
        email_content = file.read()
    
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", email_content)
    email = match.group(0) if match else "Not found"

    with open("/data/email-sender.txt", "w") as file:
        file.write(email)

    return f"Extracted email: {email}"

def task_A8():
    """
    Extract credit card number from `/data/credit-card.png` using an LLM.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": "Extract the credit card number from this image."}],
        image="/data/credit-card.png"
    )
    
    card_number = response["choices"][0]["message"]["content"].replace(" ", "")

    with open("/data/credit-card.txt", "w") as file:
        file.write(card_number)

    return "Extracted credit card number."


def task_A9():
    input_file = "/data/comments.txt"
    output_file = "/data/comments-similar.txt"

    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="comments.txt not found")

    with open(input_file, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f.readlines() if line.strip()]

    if len(comments) < 2:
        raise HTTPException(status_code=400, detail="Not enough comments for comparison")

    # Generate embeddings
    embeddings = model.encode(comments)

    # Compute cosine similarity
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    max_sim = -1
    most_similar_pair = ("", "")

    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            if similarity > max_sim:
                max_sim = similarity
                most_similar_pair = (comments[i], comments[j])

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{most_similar_pair[0]}\n{most_similar_pair[1]}")

    return {"status": "Success", "message": "Most similar comments found", "similarity": max_sim}

def task_A10():
    """
    Calculate total sales for "Gold" tickets from `/data/ticket-sales.db`.
    """
    conn = sqlite3.connect("/data/ticket-sales.db")
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type='Gold'")
    total_sales = cursor.fetchone()[0]
    
    with open("/data/ticket-sales-gold.txt", "w") as file:
        file.write(str(total_sales))

    conn.close()
    return "Calculated Gold ticket sales."



data_dir = "/data"

def ensure_data_dir():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

def fetch_api_data(url, filename):
    """Fetch data from an API and save it as JSON."""
    response = requests.get(url)
    if response.status_code == 200:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "w") as f:
            json.dump(response.json(), f)
    return filepath

def clone_modify_commit(repo_url, filename, content):
    """Clone a GitHub repo, modify a file, and commit the change."""
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join(data_dir, repo_name)
    
    if not os.path.exists(repo_path):
        git.Repo.clone_from(repo_url, repo_path)
    
    file_path = os.path.join(repo_path, filename)
    with open(file_path, "w") as f:
        f.write(content)
    
    repo = git.Repo(repo_path)
    repo.git.add(filename)
    repo.git.commit('-m', 'Automated commit')
    repo.git.push()

def execute_sql_query(db_name, query):
    """Execute an SQL query on an SQLite database."""
    db_path = os.path.join(data_dir, db_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    result = cursor.fetchall()
    conn.close()
    return result

def scrape_website(url, filename):
    """Scrape data from a website and save it."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    filepath = os.path.join(data_dir, filename)
    with open(filepath, "w") as f:
        f.write(soup.prettify())
    return filepath

def compress_image(image_filename, output_filename, quality=50):
    """Compress an image and save it."""
    input_path = os.path.join(data_dir, image_filename)
    output_path = os.path.join(data_dir, output_filename)
    image = Image.open(input_path)
    image.save(output_path, quality=quality)
    return output_path

def transcribe_audio(audio_filename):
    """Transcribe an MP3 audio file."""
    recognizer = sr.Recognizer()
    audio_path = os.path.join(data_dir, audio_filename)
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

ensure_data_dir()

@app.post("/convert_markdown")
def convert_markdown_to_html(file_path: str):
    """
    Convert a Markdown file to HTML.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Markdown file not found")

    with open(file_path, "r", encoding="utf-8") as md_file:
        md_content = md_file.read()
        html_content = markdown.markdown(md_content)

    output_path = file_path.replace(".md", ".html")
    with open(output_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)

    return {"message": "Markdown converted to HTML", "output_file": output_path}

@app.get("/filter_csv")
def filter_csv(
    file_path: str,
    column: str,
    value: Optional[str] = Query(None, description="Value to filter by"),
    min_value: Optional[float] = Query(None, description="Minimum value filter"),
    max_value: Optional[float] = Query(None, description="Maximum value filter"),
):
    """
    Filter a CSV file based on a column value or range.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="CSV file not found")

    try:
        df = pd.read_csv(file_path)
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found in CSV")

        if value is not None:
            df = df[df[column] == value]
        if min_value is not None:
            df = df[df[column] >= min_value]
        if max_value is not None:
            df = df[df[column] <= max_value]

        return json.loads(df.to_json(orient="records"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




