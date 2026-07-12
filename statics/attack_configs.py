from statics.prompt_template_en import STRIPPING_EN, TRAP_STEM_EN, TRAP_VALUE_EN
from statics.sys_prompt_en import TARGET_SYS_PROMPT_ABLATION_STEM_EN, TARGET_SYS_PROMPT_STEM_EN, TARGET_SYS_PROMPT_ABLATION_VALUE_EN, TARGET_SYS_PROMPT_VALUE_EN
from statics.format_template_en import Stripping_en, Trap_en, Target_stem_en, Target_value_en

ATTACK_CONFIGS = {
    # --- Stripping Attack ---
    ("stripping", "en", "stem"): {
        "prompt_template": STRIPPING_EN, "format_type": Stripping_en, "input_key": "meli_query"
    },
    ("stripping", "en", "value"): {
        "prompt_template": STRIPPING_EN, "format_type": Stripping_en, "input_key": "meli_query"
    },
    ("trap", "en", "stem"): {
        "prompt_template": TRAP_STEM_EN, "format_type": Trap_en, "input_key": "neutral_question"
    },
    ("trap", "en", "value"): {
        "prompt_template": TRAP_VALUE_EN, "format_type": Trap_en, "input_key": "neutral_question"
    },

    ("target", "en", "stem"): {
        "system_prompt": TARGET_SYS_PROMPT_STEM_EN, "format_type": Target_stem_en, "input_key": "trap"
    },
    ("target", "en", "value"): {
        "system_prompt": TARGET_SYS_PROMPT_VALUE_EN, "format_type": Target_value_en, "input_key": "trap"
    },

    # --- Target Attack (Ablation) ---
    ("target_ablation", "en", "stem"): {
        "system_prompt": TARGET_SYS_PROMPT_ABLATION_STEM_EN, "format_type": 'answer'
    },
    ("target_ablation", "en", "value"): {
        "system_prompt": TARGET_SYS_PROMPT_ABLATION_VALUE_EN, "format_type": 'answer'
    },

}