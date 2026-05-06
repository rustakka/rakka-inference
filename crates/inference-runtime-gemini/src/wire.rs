use serde::{Deserialize, Serialize};

use atomr_infer_core::batch::{ContentPart, ExecuteBatch, MessageContent, Role};

#[derive(Debug, Serialize)]
pub struct GenerateContentRequest<'a> {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty", rename = "safetySettings")]
    pub safety_settings: Vec<crate::config::SafetySetting>,
    #[serde(skip)]
    _model_lifetime: std::marker::PhantomData<&'a ()>,
}

#[derive(Debug, Serialize)]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Part {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: FileData,
    },
}

#[derive(Debug, Serialize)]
pub struct InlineData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Serialize)]
pub struct FileData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    #[serde(rename = "fileUri")]
    pub file_uri: String,
}

#[derive(Debug, Serialize, Default)]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topK")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty", rename = "stopSequences")]
    pub stop_sequences: Vec<String>,
}

impl GenerateContentRequest<'_> {
    pub fn from_batch<'b>(
        b: &'b ExecuteBatch,
        safety: Vec<crate::config::SafetySetting>,
    ) -> GenerateContentRequest<'b> {
        let mut system: Option<String> = None;
        let mut contents = Vec::with_capacity(b.messages.len());
        for m in &b.messages {
            if matches!(m.role, Role::System) {
                if let MessageContent::Text(t) = &m.content {
                    system = Some(system.map(|s| format!("{s}\n{t}")).unwrap_or_else(|| t.clone()));
                }
                continue;
            }
            let role = match m.role {
                Role::User | Role::Tool => "user",
                Role::Assistant => "model",
                Role::System => unreachable!(),
                // `Role` is `#[non_exhaustive]`; default unknown roles
                // to "user" so the request still goes through.
                _ => "user",
            }
            .to_string();
            let parts = match &m.content {
                MessageContent::Text(t) => vec![Part::Text { text: t.clone() }],
                MessageContent::Parts(parts) => parts.iter().map(serialize_part).collect(),
                _ => vec![Part::Text { text: String::new() }],
            };
            contents.push(Content { role, parts });
        }
        let system_instruction = system.map(|t| Content {
            role: "system".into(),
            parts: vec![Part::Text { text: t }],
        });
        GenerateContentRequest {
            contents,
            system_instruction,
            generation_config: Some(GenerationConfig {
                temperature: b.sampling.temperature,
                top_p: b.sampling.top_p,
                top_k: b.sampling.top_k,
                max_output_tokens: b.sampling.max_tokens,
                stop_sequences: b.sampling.stop.clone(),
            }),
            safety_settings: safety,
            _model_lifetime: std::marker::PhantomData,
        }
    }
}

fn serialize_part(p: &ContentPart) -> Part {
    match p {
        ContentPart::Text { text } => Part::Text { text: text.clone() },
        ContentPart::ImageBase64 { mime, data } => Part::InlineData {
            inline_data: InlineData {
                mime_type: mime.clone(),
                data: data.clone(),
            },
        },
        ContentPart::ImageUrl { url } => Part::FileData {
            file_data: FileData {
                mime_type: "image/jpeg".into(),
                file_uri: url.clone(),
            },
        },
        // Forward-compat: drop unknown variants.
        _ => Part::Text { text: String::new() },
    }
}

// ---- response -------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(default, rename = "usageMetadata")]
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct Candidate {
    #[serde(default)]
    pub content: Option<ResponseContent>,
    #[serde(default, rename = "finishReason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseContent {
    #[serde(default)]
    pub parts: Vec<ResponsePart>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsePart {
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
pub struct UsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    pub prompt_token_count: u32,
    #[serde(default, rename = "candidatesTokenCount")]
    pub candidates_token_count: u32,
    #[serde(default, rename = "cachedContentTokenCount")]
    pub cached_content_token_count: u32,
}
