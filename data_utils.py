import re
import os
import numpy as np
import contractions
import os.path as osp
from bs4 import BeautifulSoup
from datasets import DatasetDict, ClassLabel, Dataset

def clean_html_metadata(text, window_size=5, threshold=0.4):
    """
    Remove HTML tags, metadata attributes, and technical noise from text.
    Handles incomplete HTML fragments and metadata pollution from scraped content.
    Uses context-aware chain removal - removes sequences of words when technical terms
    appear in proximity (within a sliding window), avoiding removal of legitimate words
    that appear in isolation.
    
    Args:
        text: Input text string potentially containing HTML/metadata
        window_size: Size of sliding window for context analysis (default: 5)
        threshold: Minimum proportion of technical terms in window to mark word for removal (default: 0.4)
        
    Returns:
        Cleaned text with HTML and metadata removed
    """
    # First, try to parse with BeautifulSoup to remove complete HTML tags
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text(separator=' ')
    
    # Remove URLs and web-related patterns first (these are always noise)
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    text = re.sub(r'\bhttp\s+[^\s]*', '', text)
    
    # Remove complete technical metadata blocks (multi-word sequences that are always noise)
    technical_blocks = [
        # HTML/CSS/XML metadata
        r'\bencoding\s+\w+(?:\s+locale\s+\w+)?',
        r'\blocale\s+\w{2}(?:[-_]\w{2})?',
        r'\bisprivate(?:blog)?\s+(?:true|false)',
        r'\bismobile\s+(?:true|false)',
        r'\bmobileclass\s+\w*',
        r'\blanguagedirection\s+(?:ltr|rtl)',
        r'\bfeedlinks\s+\w*',
        r'\blink\s+rel\s+\w+',
        r'\btype\s+application\s+[\w/]+\s+(?:xml)?',
    ]
    
    for pattern in technical_blocks:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Consolidate common technical bigrams into single tokens before splitting
    # This allows us to treat multi-word patterns as single units
    text = re.sub(r'\ba\s+href\b', 'a_href', text, flags=re.IGNORECASE)
    text = re.sub(r'\ba\s+rel\b', 'a_rel', text, flags=re.IGNORECASE)
    text = re.sub(r'\bn\s+href\b', 'n_href', text, flags=re.IGNORECASE)
    
    # Define technical/metadata words (potential noise when in technical context)
    technical_terms = {
        'html', 'css', 'xml', 'php', 'javascript', 'js',
        'a_href', 'n_href', 'a_rel', 'src', 'rel', 'alt', 'link', 'img', 'div', 'span', 'script',
        'width', 'height', 'border', 'margin', 'padding', 'px', 'em', 'data', 'medium', 'src', 'img', 'image', 'top', 'bottom'
        'style', 'class', 'type', 'text', 'application', 'background', 'color', 'serif', 'br', 'adjust', 'auto',
        'font', 'size', 'family', 'annotation', 'display', 'indline', 'new', 'roman',
        'http', 'https', 'www', 'url', 'uri',
        'true', 'false', 'null', 'target',
        'stylesheet', 'templates', 'pub', 'message', 'board',
        'vol', 'en', 'del', 'fn',
        'atom', 'ltr', 'rtl', 'utf',
        'titled', 'title', 'smarty', 'jdelivery',
        'twitter', 'provokingbeauty'
    }
    
    # Context-aware chain removal using sliding window
    words = text.split()
    # window_size and threshold are now function parameters
    
    # Mark words that should be removed based on context
    to_remove = [False] * len(words)
    
    for i in range(len(words)):
        # Get window around current word
        start = max(0, i - window_size // 2)
        end = min(len(words), i + window_size // 2 + 1)
        window = words[start:end]
        
        # Count technical terms in window
        tech_count = sum(1 for w in window if w.lower().strip('.,!?;:') in technical_terms)
        
        # If current word is technical AND surrounded by technical terms, mark for removal
        current_word = words[i].lower().strip('.,!?;:')
        if current_word in technical_terms:
            if tech_count >= threshold * len(window):
                to_remove[i] = True
        # Also remove non-technical words if they're surrounded by many technical terms
        elif tech_count >= 0.6 * len(window):  # Higher threshold for non-technical words
            to_remove[i] = True
    
    # Expand removal to include orphaned single words between removed sequences
    for i in range(1, len(to_remove) - 1):
        if not to_remove[i] and to_remove[i-1] and to_remove[i+1]:
            # Single word sandwiched between removed words - likely part of technical sequence
            to_remove[i] = True
    
    # Clean up trailing technical fragments
    # Remove any remaining technical terms at the end of the text
    # BUT: Don't remove if it's just a single word (likely legitimate text)
    # EXCEPT: Always remove consolidated bigrams like a_href, n_href (these are always noise)
    always_remove_terms = {'a_href', 'n_href', 'a_rel'}
    
    i = len(to_remove) - 1
    trailing_count = 0
    trailing_words = []
    while i >= 0:
        current_word = words[i].lower().strip('.,!?;:')
        if current_word in technical_terms:
            trailing_count += 1
            trailing_words.append(current_word)
            i -= 1
        else:
            break  # Stop when we hit a non-technical word
    
    # Mark for removal if:
    # 1. There are 2 or more trailing technical words, OR
    # 2. The single trailing word is in always_remove_terms (consolidated bigrams)
    if trailing_count >= 2 or (trailing_count == 1 and trailing_words[0] in always_remove_terms):
        for i in range(len(to_remove) - trailing_count, len(to_remove)):
            to_remove[i] = True
    
    # Reconstruct text without marked words
    cleaned_words = [words[i] for i in range(len(words)) if not to_remove[i]]
    text = ' '.join(cleaned_words)
    
    # Remove any remaining unigrams containing 'href'
    text = re.sub(r'\b\w*href\w*\b', '', text, flags=re.IGNORECASE)
    
    # Clean up: remove multiple spaces and trim
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def fix_missing_apostrophes(text):
    """
    Fix contractions that are missing apostrophes by adding them back.
    Handles patterns like "word s" → "word's", "i m" → "i'm", etc.
    
    Args:
        text: Input text string
        
    Returns:
        Text with apostrophes restored to contractions
    """
    text_lower = text.lower()
    
    # Contraction patterns to fix
    contraction_fixes = [
        # General 's contraction (possessive or "is") - matches any word + " s"
        (r'\b([a-z]+)\s+s\b', r"\1's"),
        
        # 'm contractions
        (r'\bi\s+m\b', "i'm"),
        
        # 're contractions  
        (r'\b(you|we|they|he|she|it|that|what|who|there)\s+re\b', r"\1're"),
        
        # 'll contractions
        (r'\b(i|you|he|she|we|they|it|that|this|there|what|who)\s+ll\b', r"\1'll"),
        
        # 've contractions
        (r'\b(i|you|we|they|could|should|would|might|must)\s+ve\b', r"\1've"),
        
        # 'd contractions
        (r'\b(i|you|he|she|we|they|it|that|there|what|who)\s+d\b', r"\1'd"),
        
        # n't contractions - match any auxiliary verb + " t"
        (r'\b(do|does|did|is|are|was|were|has|have|had|wo|ca|could|would|should|must|might|ought)n?\s+t\b', r"\1n't"),
    ]
    
    for pattern, replacement in contraction_fixes:
        text_lower = re.sub(pattern, replacement, text_lower)
    
    return text_lower

def normalize_slang_text(text):
    """
    Normalize slang, abbreviations, and contractions in text.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    # Convert to lowercase for processing
    text_lower = text.lower()
    
    # # First pass: Fix contractions missing apostrophes
    # text_lower = fix_missing_apostrophes(text_lower)
    
    # Dictionary of common slang/abbreviations with regex patterns
    slang_map = {        
        # Common combinations
        r'\bb\s+c\b': 'because',  # "b c" with space
        r'\bbc\b': 'because',
        r'\bbcuz\b': 'because',
        r'\bcuz\b': 'because',
        r'\bcoz\b': 'because',
        r'\bbcos\b': 'because',
        
        # Common shortcuts
        r'\bpls\b': 'please',
        r'\bplz\b': 'please',
        r'\bthx\b': 'thanks',
        r'\bthnx\b': 'thanks',
        r'\bthanx\b': 'thanks',
        r'\bidk\b': 'i do not know',
        r'\bidgaf\b': 'i do not care',
        r'\bic\b': 'i see',
        
        # Numbers as words
        r'\b2\b': 'to',
        r'\b4\b': 'for',
        r'\b8\b': 'ate',
        r'\bb4\b': 'before',
        r'\b2day\b': 'today',
        r'\b2morrow\b': 'tomorrow',
        r'\b2nite\b': 'tonight',
        
        # Internet slang
        r'\blol\b': 'laughing out loud',
        r'\blomg\b': 'oh my god',
        r'\bomfg\b': 'oh my god',
        r'\bwtf\b': 'what the',
        r'\bbtw\b': 'by the way',
        r'\bfyi\b': 'for your information',
        r'\basap\b': 'as soon as possible',
        r'\bimy\b': 'i miss you',
        r'\bily\b': 'i love you',
        r'\btbh\b': 'to be honest',
        r'\bimo\b': 'in my opinion',
        r'\bimho\b': 'in my humble opinion',
        r'\bbrb\b': 'be right back',
        r'\bafk\b': 'away from keyboard',
        r'\bgtg\b': 'got to go',
        r'\bg2g\b': 'got to go',
        r'\bomw\b': 'on my way',
        r'\bsmh\b': 'shaking my head',
        r'\bwyd\b': 'what are you doing',
        r'\bhbu\b': 'how about you',
        r'\bnvm\b': 'never mind',
        
        # Single letters (word boundaries to avoid matching within words)
        r'\bc\b': 'see',
        r'\bu\b': 'you',
        r'\bur\b': 'your',
        r'\br\b': 'are',
        r'\by\b': 'why',
        r'\bn\b': 'and',
        r'\bw\b': 'with',
        r'\bwo\b': 'without',
    }
    
    # Apply slang replacements
    for pattern, replacement in slang_map.items():
        text_lower = re.sub(pattern, replacement, text_lower)
    
    # # Apply contractions expansion
    # text_expanded = contractions.fix(text_lower)
    
    return text_lower

def normalize_repeated_chars(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

def fix_contructions(text):
    text = fix_missing_apostrophes(text)
    text = contractions.fix(text)
    return text

def clean_sample(text):
    text = clean_html_metadata(text, window_size=5, threshold=0.35)
    text = fix_missing_apostrophes(text)
    text = normalize_slang_text(text=text)
    text = contractions.fix(text)
    text = normalize_repeated_chars(text)
    return text

def clean_text(dataset):
    text = dataset["text"]
    text = clean_sample(text)
    return {"text": text}

def remove_duplicates(dataset_dict):
    cleaned_splits = {}

    for split, dataset in dataset_dict.items():
        df = dataset.to_pandas()

        # Remove all duplicates (drop every occurrence)
        df_clean = df[~df.duplicated(subset=["text"], keep=False)]

        cleaned_splits[split] = Dataset.from_pandas(
            df_clean, preserve_index=False
        )

    return DatasetDict(cleaned_splits)

def filter_short_texts(dataset_dict, min_words=3):
    """
    Filter out texts that are shorter than a minimum number of words.
    
    Args:
        dataset_dict: DatasetDict to filter
        min_words: Minimum number of words required (default: 3)
    
    Returns:
        DatasetDict: Filtered dataset with short texts removed
    """
    filtered_splits = {}
    
    for split, dataset in dataset_dict.items():
        # Filter function to keep only texts with >= min_words
        def has_min_words(example):
            word_count = len(example['text'].split())
            return word_count >= min_words
        
        filtered_splits[split] = dataset.filter(has_min_words)
    
    return DatasetDict(filtered_splits)


def preprocess(dataset, tokenizer, validation_size=0.2, random_state=42):
    """
    Preprocess dataset by adding token length column and creating stratified train/validation splits.
    
    Args:
        dataset: DatasetDict from load_csv_to_dataset
        tokenizer: HuggingFace tokenizer
        validation_size: Proportion of training data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        DatasetDict: Processed dataset with 'train', 'validation', and 'test' splits
    """
    dataset = remove_duplicates(dataset)
    dataset = dataset.map(clean_text)
    dataset = filter_short_texts(dataset, min_words=2)  # Remove texts with < 2 words
    
    # Add length column to all splits
    def add_length(examples):
        examples['length'] = [len(tokenizer.encode(text)) for text in examples['text']]
        return examples
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    dataset = dataset.map(add_length, batched=True)
    
    # Create stratified train/validation split based on label and length bins
    train_dataset = dataset['train']
    
    # Calculate decile boundaries for length
    all_lengths = train_dataset['length']
    deciles = np.percentile(all_lengths, np.arange(0, 101, 10))
    
    # Add a combined stratification column (label + length decile)
    def add_strat_column(examples):
        strat_keys = []
        for label, length in zip(examples['label'], examples['length']):
            # Assign to decile bin (0-9)
            length_bin = np.digitize(length, deciles[1:-1])  # Exclude 0% and 100%
            # ClassLabel stores values as integers internally
            strat_key = f"{label}_{length_bin}"
            strat_keys.append(strat_key)
        examples['strat_key'] = strat_keys
        return examples
    
    train_dataset = train_dataset.map(add_strat_column, batched=True)
    
    # Convert strat_key to ClassLabel for stratification
    unique_strat_keys = sorted(set(train_dataset['strat_key']))
    train_dataset = train_dataset.cast_column(
        'strat_key',
        ClassLabel(names=unique_strat_keys)
    )
    
    # Perform stratified split
    split_dataset = train_dataset.train_test_split(
        test_size=validation_size,
        stratify_by_column='strat_key',
        seed=random_state
    )
    
    # Remove the temporary stratification column
    split_dataset['train'] = split_dataset['train'].remove_columns(['strat_key'])
    split_dataset['validation'] = split_dataset['test'].remove_columns(['strat_key'])
    
    # Create final DatasetDict with train, validation, and test
    processed_dataset = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['validation'],
        'test': dataset['test']
    })
    
    processed_dataset = processed_dataset.map(preprocess_function, batched=True)
    return processed_dataset


############################## Visulaization Utils ##############################
def highlight_text_diff(string1, string2):
    """
    Highlight differences between two strings with colors.
    
    Args:
        string1: Original string
        string2: Modified string
    
    Returns:
        tuple: (colored_string1, colored_string2) where:
            - colored_string1: string1 with deletions in red
            - colored_string2: string2 with additions in green
    """
    import difflib
    
    # ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    # Split strings into words for better diff
    words1 = string1.split()
    words2 = string2.split()
    
    # Use difflib to get the differences
    diff = difflib.SequenceMatcher(None, words1, words2)
    
    colored_string1_parts = []
    colored_string2_parts = []
    
    for tag, i1, i2, j1, j2 in diff.get_opcodes():
        if tag == 'equal':
            # Words are the same in both strings
            colored_string1_parts.extend(words1[i1:i2])
            colored_string2_parts.extend(words2[j1:j2])
        elif tag == 'delete':
            # Words deleted from string1 (not in string2) - color red in string1
            for word in words1[i1:i2]:
                colored_string1_parts.append(f"{RED}{word}{RESET}")
        elif tag == 'insert':
            # Words added to string2 (not in string1) - color green in string2
            for word in words2[j1:j2]:
                colored_string2_parts.append(f"{GREEN}{word}{RESET}")
        elif tag == 'replace':
            # Words replaced - show as deletion in string1 (red) and insertion in string2 (green)
            for word in words1[i1:i2]:
                colored_string1_parts.append(f"{RED}{word}{RESET}")
            for word in words2[j1:j2]:
                colored_string2_parts.append(f"{GREEN}{word}{RESET}")
    
    return ' '.join(colored_string1_parts), ' '.join(colored_string2_parts)

   
def show_diff(dataset, n_cases=5, fname=None):
    df = dataset["train"].to_pandas()
    df = df[df["is_cleaned"]]
    df = df.sample(n=n_cases)
    diffs = []
    for idx, (t1, t2) in enumerate(zip(df.text.tolist(), df.cleaned.tolist()), 1):
        s1, s2 = highlight_text_diff(t1, t2)
        p = f"{idx}) {s1}\n"
        p = p+ ' '*(len(str({idx}))) + f"{s2}"
        print(p)
        if idx == n_cases:
            break
        diffs.append((t1, t2))
    if fname is not None:
        fpath = f"examples/data_cleaning/{fname}"
        dirname = osp.dirname(fpath)
        os.makedirs(dirname, exist_ok=True)
        with open(fpath, "w") as f:
            for d in diffs:
                f.write("\n".join(d)+"\n")
    return

def show_duplicates(dataset):
    df = dataset["train"].to_pandas()
    duplicates = df[df.duplicated(subset=["text"], keep=False)]

    idx2label, label2idx = load_label_mapping()
    titles = ["Labels", "Tweet"]
    max_tweet_len = 80
    max_label_len = 15
    print(f"{titles[0]:^{max_label_len}} |{titles[1]:^{max_tweet_len}}")
    print("-"*(max_label_len+max_tweet_len+3))
    ctr = 0
    for t, tdf in duplicates.groupby("text"):
        if True: #len(t) <= max_tweet_len:
            labels = "({})".format('/'.join([idx2label[idx] for idx in tdf.label.tolist()]))
            print(f"{labels:<{max_label_len}} | {t}")
            ctr += 1
            if ctr == 5:
                break