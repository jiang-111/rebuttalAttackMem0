from pathlib import Path
import os, argparse, multiprocessing
from methods.trap import create_attack
import utils.file_utils as file_utils
from utils.attack_memory import (
    AttackMemory,
    AttackMemoryConfig,
    format_strategy_guidance,
    layer_strategy_cards,
    load_strategy_cards,
    merge_strategy_layers,
)
from instructor.utils import extract_json_from_codeblock as original_extract
from statics.configs import GPTAGENTBASEURL
def attack(
    attack_name: str,
    dataset_name: str,
    dataset_root:str,
    save_root = 'result',
    save_name = 'result',
    save_input = False,
    victim_llm: str = 'gpt-4o',
    base_url: str = None,
    input_key: str = None,
    concurrent: bool = True,
    memory_manager: AttackMemory = None,
    memory_metadata: dict = None,
    strategy_guidance: str = "",
    **kwargs
):
    data_path = os.path.join(dataset_root, dataset_name)
    queries= file_utils.load_prompts(data_path)
    print(f"Loaded {len(queries)} queries")

    # 解析 attack_name (例如: stripping_stem_zh)
    parts = attack_name.split('_')
    language = parts[-1]      # zh / en
    attack_type = parts[-2]   # stem / value
    method = "_".join(parts[:-2]) # stripping / trap / target / target_ablation / judge

    # 处理 target_ablation 的特殊 input_key
    # 使用工厂函数创建攻击对象
    attacker = create_attack(
        method=method,
        victim_llm=victim_llm,
        language=language,
        attack_type=attack_type,
        base_url=base_url,
        input_key=input_key,
        strategy_guidance=strategy_guidance,
        **kwargs
    )

    # 执行批量攻击
    # all_res = attacker.batch(queries=queries, save_input=save_input, concurrent=concurrent, max_workers=multiprocessing.cpu_count())
    all_res = attacker.batch(queries=queries, save_input=save_input, concurrent=concurrent, max_workers=multiprocessing.cpu_count())
    
    filepath = Path(save_root)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    save_ok = file_utils.save_to_csv(all_res, filepath, save_name)

    if memory_manager is not None:
        run_metadata = {
            "attack_name": attack_name,
            "method": method,
            "attack_type": attack_type,
            "language": language,
            "dataset_name": dataset_name,
            "dataset_root": dataset_root,
            "save_root": str(save_root),
            "save_name": save_name,
            "save_ok": save_ok,
            "victim_llm": victim_llm,
        }
        if memory_metadata:
            run_metadata.update(memory_metadata)
        memory_manager.remember_run(all_res, run_metadata)

    return save_ok


if __name__ == "__main__":
    # 1. 初始化参数解析器
    parser = argparse.ArgumentParser(description="Run Attack Experiment CLI")

    # --- 核心变量参数 ---
    
    parser.add_argument("--victim_llm", type=str, default="llama-4", 
                        help="受害者模型名称 (例如: gpt-4o, gpt-3.5-turbo)")
    
    parser.add_argument("--api_key", type=str, default=None, required=False,
                        help="OpenAI API Key (如果不传则默认读取环境变量)")
    
    parser.add_argument("--base_url", type=str, default=GPTAGENTBASEURL, 
                        help="模型 API 的 Base URL")

    parser.add_argument("--dataset_pre", type=str, choices=['stem', 'value', 'all'], default="stem",
                        help="数据集前缀，用于自动生成攻击名称和数据集名称")
    
    parser.add_argument("--attack", type=str, default="target",
                        help="攻击名称，若提供则覆盖自动生成的名称")
    
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], default='en',
                        help="语言选择，用于自动生成攻击名称")
    # --- 路径参数 ---
    parser.add_argument("--dataset_root", type=str, default="data", 
                        help="数据集根目录")
    
    parser.add_argument("--dataset_name", type=str, default="ablation_stem_en_150.csv",
                        help="数据集名称，若提供则覆盖自动生成的名称")
    
    parser.add_argument("--input_key", type=str, default=None,
                        help="输入的关键字段名称")
    
    parser.add_argument("--save_root", type=str, default="result_150",
                        help="结果保存根目录，若提供则覆盖自动生成的路径")
    
    parser.add_argument("--save_name", type=str, default="trap",
                        help="结果保存文件名")

    parser.add_argument("--save_input", nargs='+', default=['trap','meli_query'],
                    help="输入多个需要保存的字段，用空格隔开")
    
    # --- 开关参数 ---
    parser.add_argument("--no_concurrent", action="store_true", 
                        help="是否关闭并发 (默认开启并发，加上此参数则关闭)")

    # --- Mem0 Memory 参数 ---
    parser.add_argument("--memory", action="store_true",
                        help="启用 mem0 经验记忆：运行前检索相似经验，运行后写入摘要")

    parser.add_argument("--memory_backend", type=str, choices=["oss", "platform"],
                        default=os.getenv("MEM0_BACKEND", "oss"),
                        help="mem0 后端：oss 使用本地 Memory，platform 使用 MemoryClient")

    parser.add_argument("--memory_user_id", type=str,
                        default=os.getenv("MEM0_USER_ID", "rebuttal_attack"),
                        help="mem0 user_id，用于隔离本项目经验")

    parser.add_argument("--memory_agent_id", type=str,
                        default=os.getenv("MEM0_AGENT_ID", "attack_memory"),
                        help="mem0 agent_id")

    parser.add_argument("--memory_app_id", type=str,
                        default=os.getenv("MEM0_APP_ID", "rebuttalAttack"),
                        help="mem0 app_id")

    parser.add_argument("--memory_run_id", type=str, default=None,
                        help="mem0 run_id；不填则自动生成")

    parser.add_argument("--memory_config", type=str, default=None,
                        help="mem0 OSS 配置文件路径，支持 JSON/YAML")

    parser.add_argument("--memory_top_k", type=int, default=5,
                        help="运行前检索多少条历史经验")

    parser.add_argument("--memory_item_limit", type=int, default=0,
                        help="额外写入多少条逐样本 digest；默认只写 run summary")

    parser.add_argument("--memory_dry_run", action="store_true",
                        help="只打印将写入/检索的 mem0 payload，不调用 mem0")

    parser.add_argument("--memory_cards", type=str, default=None,
                        help="本地三层策略卡 JSONL；可单独使用，也可与 Mem0 检索结果合并")

    parser.add_argument("--memory_guidance_top_k", type=int, default=3,
                        help="每一层最多检索/注入多少条策略")

    parser.add_argument("--memory_guide_target", action="store_true",
                        help="也把记忆注入 target/target_ablation；默认仅指导攻击提示生成，避免污染受害模型评测")
    
    # 2. 解析参数
    args = parser.parse_args()
    # args.no_concurrent = True

    # 3. 设置环境变量
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
        os.environ['GEMINI_API_KEY'] = args.api_key
        os.environ["OPENROUTER_API_KEY"] = args.api_key
        
    elif 'OPENAI_API_KEY' not in os.environ:
        print("⚠️ Warning: No API Key provided via args or environment variables.")

    # 4. 逻辑变量构建 (保留您原本的 f-string 逻辑)
    dataset_pre = args.dataset_pre
    
    # 自动构建 attack_name 和 dataset_name
    attack_name = f"{args.attack}_{dataset_pre}_{args.lang}"
    memory_metadata = {
        "attack_name": attack_name,
        "method": args.attack,
        "attack_type": dataset_pre,
        "language": args.lang,
        "dataset_name": args.dataset_name,
        "victim_llm": args.victim_llm,
    }

    memory_manager = None
    remote_layers = {"success": [], "false_positive": [], "failure": []}
    if args.memory:
        try:
            memory_manager = AttackMemory(
                AttackMemoryConfig(
                    backend=args.memory_backend,
                    user_id=args.memory_user_id,
                    agent_id=args.memory_agent_id,
                    app_id=args.memory_app_id,
                    run_id=args.memory_run_id,
                    config_path=args.memory_config,
                    top_k=args.memory_top_k,
                    item_limit=args.memory_item_limit,
                    dry_run=args.memory_dry_run,
                )
            )
            remote_layers = memory_manager.search_strategy_guidance(
                memory_metadata,
                top_k_per_outcome=args.memory_guidance_top_k,
            )
        except Exception as e:
            print(f"🧠 Memory disabled: {e}")

    local_layers = {"success": [], "false_positive": [], "failure": []}
    if args.memory_cards:
        try:
            local_layers = layer_strategy_cards(load_strategy_cards(args.memory_cards))
        except (OSError, ValueError) as e:
            print(f"🧠 Local strategy cards disabled: {e}")

    strategy_layers = merge_strategy_layers(remote_layers, local_layers)
    strategy_guidance = format_strategy_guidance(
        strategy_layers,
        limit_per_outcome=args.memory_guidance_top_k,
    )
    target_stage = args.attack in {"target", "target_ablation", "judge"}
    if strategy_guidance:
        print("🧠 Three-layer attack guidance:")
        print(strategy_guidance)
        if target_stage and not args.memory_guide_target:
            print("🧠 Guidance retrieved but not injected into the target stage (use --memory_guide_target to override).")
            active_strategy_guidance = ""
        else:
            active_strategy_guidance = strategy_guidance
    else:
        if args.memory or args.memory_cards:
            print("🧠 No three-layer strategy guidance retrieved.")
        active_strategy_guidance = ""

    # 自动构建 save_root
    save_root = args.save_root if args.save_root is not None else f"data/{dataset_pre}_150"
    save_path = f'{save_root}/{args.dataset_pre}/{args.victim_llm.split("/")[-1]}'
    os.makedirs(save_path, exist_ok=True)
    
    print("="*40)
    print(f"🚀 Launching Attack Job")
    print(f"📍 Attack Name : {attack_name}")
    print(f"📄 Dataset     : {os.path.join(args.dataset_root, args.dataset_name)}")
    print(f"💾 Save Root   : {save_path}")
    print(f"🤖 Victim LLM  : {args.victim_llm}")
    print(f"⚡ Concurrent  : {not args.no_concurrent}")
    print("="*40)

    attack(
        attack_name=attack_name,
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        save_root=save_path,
        save_name=args.save_name,
        save_input=args.save_input,
        victim_llm=args.victim_llm,
        base_url=args.base_url,
        input_key=args.input_key,
        concurrent=not args.no_concurrent,  #如果不加 --no_concurrent，这里就是 True
        memory_manager=memory_manager,
        memory_metadata=memory_metadata,
        strategy_guidance=active_strategy_guidance,
    )
