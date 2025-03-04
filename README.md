# Genie-setup

This repository contains scripts for running the GENIE medical language model on clinical notes. The setup is optimized for Columbia University's High-Performance Computing (HPC) environments.

## Setup Instructions

### 1. Request a GPU Node

First, request a GPU node on your HPC cluster:

```bash
srun --pty -t 0-01:00 --gres=gpu:1 --mem=64g -A 5sigma /bin/bash
```

This command requests:
- An interactive session (`--pty`)
- 1 hour of runtime (`-t 0-01:00`)
- 1 GPU (`--gres=gpu:1`)
- 64GB of memory (`--mem=64g`)
- Using the 5sigma(biostat) account (`-A 5sigma`)

### 2. Install Miniconda

If you don't have conda installed, you can install Miniconda:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Initialize conda in your shell:

```bash
~/miniconda3/bin/conda init bash
```

Close and reopen your terminal, or source your bashrc:

```bash
source ~/.bashrc
```

For more detailed instructions, visit: [Install Miniconda](https://waylonwalker.com/install-miniconda/)

### 3. Set Up Conda Environment

Create and activate a new conda environment:

```bash
conda create -n vllm_inference python=3.10
conda activate vllm_inference
```

Install the required packages using the vllm_requirements.txt file:

```bash
pip install -r vllm_requirements.txt
```

### 4. Configure Hugging Face Cache

To avoid downloading the model each time, set up a persistent cache directory:

```bash
# Create a personal cache directory
mkdir -p /insomnia001/depts/5sigma/users/$(whoami)/cache

# Set the HF_HOME environment variable
export HF_HOME=/insomnia001/depts/5sigma/users/$(whoami)/cache
```

Add this to your `.bashrc` to make it permanent:

```bash
echo 'export HF_HOME=/insomnia001/depts/5sigma/users/'$(whoami)'/cache' >> ~/.bashrc
```

### 5. Running the Scripts

The repository contains the following script:
- `test_mimic.py`: Processes clinical notes and generates analysis using the GENIE model

To run the analysis:

```bash
python test_mimic.py
```

### 6. Handling Long Clinical Notes

If your clinical notes are too long, you can chunk them before processing. It's important to use semantic chunking rather than arbitrary character-based chunking to preserve the meaning and context of the clinical information.

Here's an example of how to implement semantic chunking:

```python
def semantic_chunk_note(note, max_length=4000, overlap=200):
    """
    Split a long note into semantically meaningful chunks.
    
    Args:
        note (str): The clinical note to chunk
        max_length (int): Maximum length of each chunk
        overlap (int): Number of characters to overlap between chunks for context
    
    Returns:
        list: List of chunked notes
    """
    # If note is shorter than max_length, return it as is
    if len(note) <= max_length:
        return [note]
    
    chunks = []
    start = 0
    
    while start < len(note):
        # Determine end position (either max_length or end of note)
        end = min(start + max_length, len(note))
        
        # If we're not at the end of the note, try to find a good breaking point
        if end < len(note):
            # Look for paragraph breaks, periods, or other natural breaking points
            # Try to find paragraph break first
            paragraph_break = note.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + max_length // 2:
                end = paragraph_break + 2  # Include the newlines
            else:
                # Try to find sentence end (period followed by space or newline)
                sentence_end = max(
                    note.rfind('. ', start + max_length // 2, end),
                    note.rfind('.\n', start + max_length // 2, end)
                )
                if sentence_end != -1:
                    end = sentence_end + 2  # Include the period and space/newline
        
        # Add the chunk
        chunks.append(note[start:end])
        
        # Move start position for next chunk, with overlap for context
        start = max(start, end - overlap) if end < len(note) else end
    
    return chunks

# Example usage in your code
chunked_notes = []
for note in clinical_notes:
    chunks = semantic_chunk_note(note)
    chunked_notes.extend(chunks)

# Process chunked_notes instead of clinical_notes
```

This semantic chunking approach:
- Tries to break notes at natural boundaries like paragraph breaks or sentence endings
- Maintains overlap between chunks to preserve context
- Ensures that medical concepts aren't arbitrarily split in the middle

## Output

The script generates a file called `genie_analysis_results.txt` containing the analysis of each clinical note. The output is formatted as JSON for better readability.

## Troubleshooting

- **Out of memory errors**: Try reducing the batch size or chunking your notes into smaller pieces
- **Model download issues**: Check your internet connection and ensure your HF_HOME directory has enough space
- **CUDA errors**: Make sure you have requested a GPU node and that your CUDA drivers are compatible with PyTorch

## References

- [GENIE Model](https://huggingface.co/THUMedInfo/GENIE_en_8b)
- [vLLM Documentation](https://github.com/vllm-project/vllm)