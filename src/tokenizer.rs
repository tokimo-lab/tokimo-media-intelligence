/// BERT `WordPiece` tokenizer for Chinese-CLIP.
/// Vocabulary is embedded at compile time from config/vocab.txt (21128 tokens).
use std::collections::HashMap;

static VOCAB_TEXT: &str = include_str!("../config/vocab.txt");

pub struct BertTokenizer {
    vocab: HashMap<String, i64>,
    cls_id: i64,
    sep_id: i64,
}

impl BertTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        for (i, line) in VOCAB_TEXT.lines().enumerate() {
            let token = line.trim();
            if !token.is_empty() {
                vocab.insert(token.to_string(), i as i64);
            }
        }
        let cls_id = vocab.get("[CLS]").copied().unwrap_or(101);
        let sep_id = vocab.get("[SEP]").copied().unwrap_or(102);
        Self { vocab, cls_id, sep_id }
    }

    /// Tokenize text and return padded token IDs of length `context_length`.
    pub fn encode(&self, text: &str, context_length: usize) -> Vec<i64> {
        let tokens = self.tokenize(text);
        let token_ids = self.convert_tokens_to_ids(&tokens);

        // [CLS] + tokens[..context_length-2] + [SEP]
        let max_tokens = context_length.saturating_sub(2);
        let truncated = &token_ids[..token_ids.len().min(max_tokens)];

        let mut result = vec![0i64; context_length];
        result[0] = self.cls_id;
        for (i, &id) in truncated.iter().enumerate() {
            result[i + 1] = id;
        }
        result[truncated.len() + 1] = self.sep_id;
        result
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let text = text.to_lowercase();
        let cleaned = clean_text(&text);
        let with_cjk_spaces = tokenize_chinese_chars(&cleaned);

        let mut tokens = Vec::new();
        for word in with_cjk_spaces.split_whitespace() {
            let stripped = strip_accents(word);
            for piece in split_on_punctuation(&stripped) {
                // WordPiece
                tokens.extend(self.wordpiece(&piece));
            }
        }
        tokens
    }

    fn wordpiece(&self, token: &str) -> Vec<String> {
        if token.len() > 200 {
            return vec!["[UNK]".to_string()];
        }

        let chars: Vec<char> = token.chars().collect();
        let mut output = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;

            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let candidate = if start > 0 { format!("##{substr}") } else { substr };
                if self.vocab.contains_key(&candidate) {
                    found = Some(candidate);
                    break;
                }
                end -= 1;
            }

            if let Some(tok) = found {
                output.push(tok);
                start = end;
            } else {
                output.push("[UNK]".to_string());
                break;
            }
        }
        output
    }

    fn convert_tokens_to_ids(&self, tokens: &[String]) -> Vec<i64> {
        tokens
            .iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(100)) // 100 = [UNK]
            .collect()
    }
}

fn clean_text(text: &str) -> String {
    text.chars()
        .filter_map(|c| {
            if c == '\0' || c == '\u{fffd}' || is_control(c) {
                None
            } else if is_whitespace(c) {
                Some(' ')
            } else {
                Some(c)
            }
        })
        .collect()
}

fn tokenize_chinese_chars(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    for c in text.chars() {
        if is_chinese_char(c) {
            out.push(' ');
            out.push(c);
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

fn strip_accents(text: &str) -> String {
    // Simplified accent stripping: iterate chars and remove combining marks
    text.chars().filter(|c| !is_combining_mark(*c)).collect()
}

fn is_combining_mark(c: char) -> bool {
    let cp = c as u32;
    (0x0300..=0x036F).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
        || (0xFE20..=0xFE2F).contains(&cp)
}

fn split_on_punctuation(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if is_punctuation(c) {
            if !current.is_empty() {
                result.push(std::mem::take(&mut current));
            }
            result.push(c.to_string());
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

fn is_chinese_char(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r') || c.is_whitespace()
}

fn is_control(c: char) -> bool {
    if matches!(c, '\t' | '\n' | '\r') {
        return false;
    }
    c.is_control()
}

fn is_punctuation(c: char) -> bool {
    let cp = c as u32;
    if (33..=47).contains(&cp) || (58..=64).contains(&cp) || (91..=96).contains(&cp) || (123..=126).contains(&cp) {
        return true;
    }
    // Unicode punctuation categories
    c.is_ascii_punctuation() || matches!(unicode_general_category_broad(c), 'P' | 'S')
}

fn unicode_general_category_broad(c: char) -> char {
    // Simplified: check if it's punctuation by common Unicode ranges
    if c.is_alphanumeric() || c.is_whitespace() {
        'L'
    } else {
        'P'
    }
}
