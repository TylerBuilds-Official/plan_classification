"""Standard construction drawing discipline patterns"""

# Maps discipline name → regex pattern for sheet number matching
# Prefixes follow CSI/NCS conventions with flexibility for separators
CATEGORY_PATTERNS = {
    'Architectural':   r'\bA[D]?\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Civil':           r'\bC[A]?\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Electrical':      r'\bE[LP]?\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Fire Protection': r'\bFP\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Food Service':    r'\bFS\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'General':         r'\bG\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Landscape':       r'\bL[AS]?\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Life Safety':     r'\bLS\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Mechanical':      r'\bM[GP]?\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Plumbing':        r'\bP\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Security':        r'\bSS\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Structural':      r'\bS\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Technology':      r'\bT\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Survey':          r'\bV\s*(?:[-.\s]?\d+)+[A-Z]*\b',
}
