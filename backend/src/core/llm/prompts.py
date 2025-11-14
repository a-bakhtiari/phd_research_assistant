from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class PromptTemplate:
    """Template for LLM prompts with variables."""
    name: str
    system_prompt: str
    user_prompt: str
    variables: List[str]
    description: str = ""
    
    def format(self, **kwargs) -> List[Dict[str, str]]:
        """Format the prompt template with provided variables."""
        # Check if all required variables are provided
        missing_vars = [var for var in self.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Format prompts
        formatted_system = self.system_prompt.format(**kwargs)
        formatted_user = self.user_prompt.format(**kwargs)
        
        messages = []
        if formatted_system.strip():
            messages.append({"role": "system", "content": formatted_system})
        if formatted_user.strip():
            messages.append({"role": "user", "content": formatted_user})
        
        return messages


class PromptManager:
    """Manages prompt templates for different tasks."""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing custom prompt files
        """
        self.prompts = {}
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None
        
        # Load default prompts
        self._load_default_prompts()
        
        # Load custom prompts if directory provided
        if self.prompts_dir and self.prompts_dir.exists():
            self._load_custom_prompts()
    
    def _load_default_prompts(self):
        """Load default built-in prompts."""
        
        # Paper metadata extraction prompt
        self.prompts["extract_metadata"] = PromptTemplate(
            name="extract_metadata",
            system_prompt="""You are an expert academic assistant that extracts metadata from academic papers.
Your task is to analyze the provided text (typically from the first 2-3 pages of a paper) and extract key information.

Additionally, if Semantic Scholar data is provided, you must verify if it matches the paper by comparing:
- Title similarity (allowing for reasonable variations, abbreviations, formatting differences)
- Author names (accounting for different name formats, initials vs full names, author order)
- Publication context and content alignment

You must respond with valid JSON following this exact schema:
{{
    "title": "string - full paper title",
    "authors": ["string array - all author names"],
    "year": "integer - publication year", 
    "abstract": "string - paper abstract if available",
    "journal": "string - journal/conference name if available",
    "doi": "string - DOI if available",
    "keywords": ["string array - key topics/keywords"],
    "verified_semantic_scholar_match": "boolean - true if Semantic Scholar data matches this paper",
    "use_semantic_scholar_data": "boolean - true if SS data should be used for enrichment"
}}

For verification:
- Set both verification fields to true if you're confident the Semantic Scholar result is the same paper
- Set both to false if there's reasonable doubt or clear mismatch
- Consider title variations, author formatting differences, but reject if core content differs
- If no Semantic Scholar data is provided, set both verification fields to false

Be precise and only extract information that is clearly stated in the text.""",
            user_prompt="Extract metadata from this paper text:\n\n{paper_text}\n\n{semantic_scholar_context}",
            variables=["paper_text", "semantic_scholar_context"],
            description="Extract metadata from academic paper text and verify Semantic Scholar matches"
        )
        
        # Paper summarization prompt
        self.prompts["summarize_paper"] = PromptTemplate(
            name="summarize_paper",
            system_prompt="""You are an expert academic reviewer. Create a concise but comprehensive summary of the academic paper.

Your summary should be approximately 150-200 words and cover:
1. The main research problem or question
2. The methodology or approach used
3. Key findings or contributions
4. Implications or significance

Write in an objective, academic tone suitable for a literature review.""",
            user_prompt="Summarize this academic paper:\n\nTitle: {title}\nAuthors: {authors}\n\nContent:\n{content}",
            variables=["title", "authors", "content"],
            description="Generate academic summary of a paper"
        )
        
        # Structured paper analysis prompt
        self.prompts["structured_paper_analysis"] = PromptTemplate(
            name="structured_paper_analysis",
            system_prompt="""You are an expert academic researcher and a specialist in literature analysis with exceptional attention to detail. Your task is to provide a structured, rigorous summary of the academic paper provided below. The final summary must be concise enough to fit on a single page when formatted.

Your entire output **must be a single, valid JSON object**. Do not include any text, explanations, or markdown formatting before or after the JSON object.

It is **critical** that you adhere strictly to the specified word counts for the content within each section to meet the one-page limit. The target lengths are:
* `executive_summary`: ~30 words
* `purpose_rationale_research_question`: ~125 words total
* `theory_framework`: ~50 words total
* `methodology`: ~125 words total
* `major_findings_contributions`: ~150 words total
* `study_limitations_gaps_that_remain`: ~75 words total
* `study_implications_for_research_practice_policy`: ~75 words total

For any point that requires a factual claim, represent it as an object with two keys: "tag" and "content".
- The "tag" key's value must be one of two strings: "Explicitly Stated" or "Inferred".
- The "content" key's value will be the text of the summary point, adhering to the word counts above.""",
            user_prompt="""Analyze this academic paper and provide the structured JSON summary:

{{
  "title": "[The full title of the paper]",
  "executive_summary": "[A single, concise sentence of ~30 words encapsulating the paper's core argument, method, and main finding.]",
  "purpose_rationale_research_question": [ // ~125 words total
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "[The purpose and primary goals of the research.]"
    }},
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "[The rationale: the justification for the study and why it is important.]"
    }},
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "RQ: [The central research question(s) the study addresses.]"
    }}
  ],
  "theory_framework": [ // ~50 words total
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "[Identify the primary theory, theoretical framework, or conceptual model used. If none is mentioned, state 'No explicit theoretical framework was mentioned'.]"
    }}
  ],
  "methodology": [ // ~125 words total
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "[The overall research approach and design (e.g., qualitative, quantitative, experimental).]"
    }},
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "[The key techniques, procedures, and data collection methods.]"
    }},
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "[Description of the data, sample, or population.]"
    }},
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "[Primary methods used for data analysis.]"
    }}
  ],
  "major_findings_contributions": [ // ~150 words total
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "Finding 1: [A major finding or outcome of the research.]"
    }},
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "Finding 2: [Another major finding.]"
    }},
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "Contribution: [The primary contribution(s) to the field, such as closing a gap, providing a new model, or offering new evidence.]"
    }}
  ],
  "study_limitations_gaps_that_remain": [ // ~75 words total
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "[The limitations of the study as acknowledged by the authors.]"
    }},
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "[The remaining gaps in knowledge that this study exposes or fails to address.]"
    }}
  ],
  "study_implications_for_research_practice_policy": [ // ~75 words total
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "For Research: [Implications for future academic work or studies.]"
    }},
    {{
      "tag": "Explicitly Stated or Inferred",
      "content": "For Practice/Policy: [Implications for practitioners, professionals, or policy-makers.]"
    }}
  ]
}}

---
PAPER TEXT:
{paper_text}
---""",
            variables=["paper_text"],
            description="Generate structured academic analysis with specific JSON format"
        )
        
        # Literature gap analysis prompt
        self.prompts["find_gaps"] = PromptTemplate(
            name="find_gaps",
            system_prompt="""You are an expert academic researcher conducting literature gap analysis.

Your task is to:
1. Analyze the current literature and research context
2. Identify potential gaps or underexplored areas
3. Suggest how specific papers might fill these gaps
4. Explain the relevance and potential contribution

Be specific about why each suggested paper would strengthen the literature review.""",
            user_prompt="""Research context: {research_context}

Current literature (papers already reviewed):
{current_papers}

Potential new papers found:
{candidate_papers}

Identify which papers would best fill gaps in the current literature and explain why.""",
            variables=["research_context", "current_papers", "candidate_papers"],
            description="Identify literature gaps and suggest relevant papers"
        )
        
        # Writing assistance prompt
        self.prompts["writing_assistant"] = PromptTemplate(
            name="writing_assistant",
            system_prompt="""You are an expert academic writing assistant. Help improve academic writing by:

1. Maintaining appropriate academic tone and style
2. Ensuring clarity and precision
3. Following academic writing conventions
4. Integrating relevant citations naturally
5. Improving flow and coherence

Your suggestions should be constructive and maintain the author's voice while enhancing quality.""",
            user_prompt="""Document context: {document_context}

Current text to improve:
{current_text}

Task: {task_description}

Relevant references available:
{available_references}

Please provide improved text with explanations for major changes.""",
            variables=["document_context", "current_text", "task_description", "available_references"],
            description="Provide academic writing assistance and improvements"
        )
        
        # Revision response prompt
        self.prompts["revision_helper"] = PromptTemplate(
            name="revision_helper",
            system_prompt="""You are an expert academic editor helping authors respond to peer review comments.

For each reviewer comment:
1. Analyze what the reviewer is asking for
2. Suggest 2-3 specific strategies to address the concern
3. Provide pros and cons for each strategy
4. Recommend the best approach

Be diplomatic and constructive in your suggestions.""",
            user_prompt="""Original text: {original_text}

Reviewer comment: {reviewer_comment}

Provide strategies to address this comment while maintaining the paper's integrity.""",
            variables=["original_text", "reviewer_comment"],
            description="Help respond to peer review comments"
        )
        
        # Research chat prompt
        self.prompts["research_chat"] = PromptTemplate(
            name="research_chat",
            system_prompt="""You are a knowledgeable research assistant with access to the user's paper library.

Use the provided context from their papers to answer questions accurately. Always:
1. Base answers on the provided context
2. Cite specific papers when relevant
3. Acknowledge when information is not available in the library
4. Provide helpful suggestions for further research

Be precise and scholarly in your responses.""",
            user_prompt="""Context from research library:
{context}

User question: {question}

Please answer based on the available context and suggest additional research directions if relevant.""",
            variables=["context", "question"],
            description="Answer research questions using paper library context"
        )
        
        # Paper recommendation prompt
        self.prompts["recommend_papers"] = PromptTemplate(
            name="recommend_papers",
            system_prompt="""You are an expert research librarian making paper recommendations.

Analyze the research context and current work to suggest highly relevant papers. For each recommendation:
1. Explain how it relates to the current research
2. Describe its potential contribution
3. Indicate where it might be most useful (methodology, related work, etc.)
4. Assess its importance (high/medium/low priority)

Focus on papers that would genuinely strengthen the research.""",
            user_prompt="""Current research focus: {research_focus}

Current document/project: {current_work}

Papers already in library: {existing_papers}

Available candidate papers: {candidate_papers}

Provide ranked recommendations with clear explanations.""",
            variables=["research_focus", "current_work", "existing_papers", "candidate_papers"],
            description="Recommend relevant papers for current research"
        )
        
        # Academic paper verification prompt
        self.prompts["verify_academic_paper"] = PromptTemplate(
            name="verify_academic_paper",
            system_prompt="""You are an expert academic document classifier. Your task is to determine if a PDF document is an academic research paper.

Academic research papers typically have these characteristics:
- Clear title, author(s), and institutional affiliations
- Abstract summarizing the research
- Introduction, methodology, results, and conclusion sections
- References/citations to other academic works
- Academic writing style and terminology
- Original research, experiments, or scholarly analysis
- Published in journals, conferences, or as preprints

NOT academic papers:
- News articles, blog posts, or opinion pieces
- Books (unless research monographs)
- Technical manuals or user guides
- Business reports or marketing materials
- Personal documents (CVs, forms, invoices)
- Course materials or textbooks (unless research-focused)

You must respond with valid JSON following this exact schema:
{{
    "is_academic_paper": "boolean - true if this is an academic research paper",
    "confidence": "number 0-100 - confidence level in your assessment",
    "reasoning": "string - brief explanation of your decision",
    "detected_elements": ["array of strings - academic elements found (title, abstract, authors, etc.)"],
    "missing_elements": ["array of strings - expected academic elements that are missing"]
}}

Be conservative - if you're unsure, err on the side of NOT classifying it as academic.""",
            user_prompt="Analyze this PDF content and determine if it's an academic research paper:\n\n{pdf_text}",
            variables=["pdf_text"],
            description="Verify if a PDF document is an academic research paper"
        )
    
    def _load_custom_prompts(self):
        """Load custom prompts from the prompts directory."""
        if not self.prompts_dir.exists():
            return
        
        for prompt_file in self.prompts_dir.glob("*.txt"):
            try:
                content = prompt_file.read_text(encoding="utf-8")
                
                # Parse prompt file (simple format: SYSTEM:...\nUSER:...\nVARIABLES:...)
                sections = {}
                current_section = None
                current_content = []
                
                for line in content.split('\n'):
                    if line.startswith(('SYSTEM:', 'USER:', 'VARIABLES:', 'DESCRIPTION:')):
                        if current_section:
                            sections[current_section] = '\n'.join(current_content).strip()
                        current_section = line.split(':')[0].lower()
                        current_content = [':'.join(line.split(':')[1:]).strip()]
                    else:
                        current_content.append(line)
                
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Create prompt template
                if 'system' in sections and 'user' in sections:
                    variables = []
                    if 'variables' in sections:
                        variables = [v.strip() for v in sections['variables'].split(',')]
                    
                    prompt_name = prompt_file.stem
                    self.prompts[prompt_name] = PromptTemplate(
                        name=prompt_name,
                        system_prompt=sections['system'],
                        user_prompt=sections['user'],
                        variables=variables,
                        description=sections.get('description', f"Custom prompt: {prompt_name}")
                    )
                    
            except Exception as e:
                print(f"Warning: Failed to load custom prompt {prompt_file}: {e}")
    
    def get_prompt(self, name: str) -> PromptTemplate:
        """Get prompt template by name."""
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found. Available: {list(self.prompts.keys())}")
        return self.prompts[name]
    
    def list_prompts(self) -> List[str]:
        """List available prompt names."""
        return list(self.prompts.keys())
    
    def format_prompt(self, name: str, **kwargs) -> List[Dict[str, str]]:
        """Format a prompt template with variables."""
        prompt = self.get_prompt(name)
        return prompt.format(**kwargs)
    
    def add_custom_prompt(self, 
                         name: str, 
                         system_prompt: str, 
                         user_prompt: str, 
                         variables: List[str],
                         description: str = ""):
        """Add a custom prompt template."""
        self.prompts[name] = PromptTemplate(
            name=name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            variables=variables,
            description=description
        )
    
    def save_custom_prompt(self, name: str, filename: str = None):
        """Save a prompt template to file."""
        if not self.prompts_dir:
            raise ValueError("No prompts directory configured")
        
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found")
        
        prompt = self.prompts[name]
        filename = filename or f"{name}.txt"
        filepath = self.prompts_dir / filename
        
        content = f"""SYSTEM: {prompt.system_prompt}

USER: {prompt.user_prompt}

VARIABLES: {', '.join(prompt.variables)}

DESCRIPTION: {prompt.description}
"""
        
        filepath.write_text(content, encoding="utf-8")


# JSON schemas for structured responses
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "year": {"type": "integer"},
        "abstract": {"type": "string"},
        "journal": {"type": "string"},
        "doi": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        # Temporary verification fields (not saved to database)
        "verified_semantic_scholar_match": {"type": "boolean"},
        "use_semantic_scholar_data": {"type": "boolean"}
    },
    "required": ["title", "authors", "verified_semantic_scholar_match", "use_semantic_scholar_data"]
}

RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_title": {"type": "string"},
                    "relevance_score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "reason": {"type": "string"},
                    "category": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                },
                "required": ["paper_title", "relevance_score", "reason", "priority"]
            }
        }
    },
    "required": ["recommendations"]
}

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_academic_paper": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100},
        "reasoning": {"type": "string"},
        "detected_elements": {
            "type": "array",
            "items": {"type": "string"}
        },
        "missing_elements": {
            "type": "array", 
            "items": {"type": "string"}
        }
    },
    "required": ["is_academic_paper", "confidence", "reasoning"]
}