import argparse
import re
import string
import requests
import json
from utils import  read_warc_file, retrieve_bad_words
from datasets import load_dataset
from typing import Set, Dict
from bs4 import BeautifulSoup as bs
import html2text

def retrieve_bad_words() -> set[str]:
    """Helper function - that reads a list of bad words from a file and returns them as a set.
    Returns:
        Set[str]: A set containing lowercase bad words.
    """
    with open('./bad_word_list.txt', 'r') as file:
        records = file.read().strip().split('\n')
        bad_words = [record.lower() for record in records]
        return set(bad_words)

def html_to_text(html: str) -> str:
    """Converts HTML content to plain text..
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Plain text extracted from HTML.
    """
    # markdown form
    # soup = bs(html, 'html.parser')
    # if not isinstance(html, str):
    #     html = html.decode(soup.original_encoding, errors='replace')
    # soup = bs(html, 'html.parser')
    # for tag in soup.find_all('a'):
    #     tag.unwrap()
    # html = str(soup)
    
    # html_cleaner = html2text.HTML2Text()
    # html_cleaner.ignore_images = True
    # html_cleaner.ignore_links = True
    # html_cleaner.ignore_tables = True
    # text = html_cleaner.handle(html)

    # text = '\n'.join([line for line in text.split('\n') if line.strip()])
    
    #no markdown
    soup = bs(html, 'html.parser')
    text = '\n'.join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
    return text


def replace_pii(text: str) -> str:
    """Masks personally identifiable information (PII) from text with the specified masking formats.
    Args: 
        text (str): Candidate text.
    Returns:
        str: Text with PII obfuscated.
    """
    pattern1 = r'\d\d\d-\d\d-\d\d\d\d'
    pattern2 = r'\+1\d{10}'
    text = re.sub(pattern1, 'XXX-XX-XXXX', text)
    text = re.sub(pattern2, '+1XXXXXXXXXX', text) 
    return text

def clean_text(text: str) -> str:
    """Removes substrings identified as low-quality according to alphanumeric, whitespace and valid document checks.  
    Args:
        text (str): document to process.
    Returns:
        str: cleaned document
    """
    paragraphs = text.split('\n')
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        if re.search(r'\w{101,}', paragraph):
            continue
        if not any(char in string.punctuation for char in paragraph):
            continue
        cleaned_paragraphs.append(paragraph)
    
    return '\n'.join(cleaned_paragraphs)
     


def heuristic_quality_filter(text: str) -> bool:
    """Rejects documents based on the presence of bad words and punctuation.
    Args:
        text (str): document to check
    Returns:
        bool: returns True if the document passes the filters, False otherwise.
    """
    # - contains no words from the bad words list located here. 
    bad_words = retrieve_bad_words()
    for word in bad_words:
        word = re.escape(word)
        if re.search(word, text, re.IGNORECASE):
            return False
    
    # - contains punctuation (use strings.punctuation for this).
    if not re.search(r'[' + re.escape(string.punctuation) + r'\s]', text):
        return False
    
    # - contains non-whitespace characters. 
    # 似乎只要有标点符号可以了啊
    if not re.search(r'[^\s]', text):
        return False
    
    # - at least 80% of characters in the document are one of: alphanumeric, punctuation, whitespace
    pattern = r'[a-zA-Z \d]'
    if len(re.findall(pattern, text)) / len(text) > 0.8:
        return True
    else:
        return False
    


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type = str,  default = 'data.warc', help = 'Specify the path for your warc file.')
    parser.add_argument('--num_records', type = int,  default=None, help = 'Specify the number of records you want to parse (only used for debugging with smaller sets)')
    args = parser.parse_args()
    sum = 0
    count = 0
    if args.fname:
        for url, html_text in read_warc_file(args.fname, args.num_records):
            sum += 1
            text = html_to_text(html_text)
            if len(text) == 0:
                count += 1
            cleaned_text = clean_text(text)
            cleaned_nopii_text = replace_pii(cleaned_text)
            passes_check = heuristic_quality_filter(cleaned_nopii_text)
            
            if sum % 1000 == 0:
                print(url)
                print("Passes heuristic quality filter:", passes_check)
                print(cleaned_nopii_text)
                print("\n\n\n")
        # sum = 6368, deleted = 52
        print(f'total number: {sum}, deleted number: {count}')
    else:
        print("Usage: python homework.py --fname data.warc")