"""
一键运行全部分层评估
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import subprocess
import json
from datetime import datetime


def run_command(cmd: list, description: str):
    """运行命令并打印结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=str(Path(__file__).parent.parent)
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return False


def generate_report():
    """生成评估报告"""
    report_dir = Path(__file__).parent

    # 收集结果
    results = {}

    # 检索评估结果
    retrieval_path = report_dir / "retrieval_eval_result.json"
    if retrieval_path.exists():
        with open(retrieval_path, 'r', encoding='utf-8') as f:
            results['retrieval'] = json.load(f)

    # 端到端评估结果
    end2end_path = report_dir / "end2end_eval_result.json"
    if end2end_path.exists():
        with open(end2end_path, 'r', encoding='utf-8') as f:
            results['end2end'] = json.load(f)

    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {}
    }

    if 'retrieval' in results:
        r = results['retrieval']['overall']
        report['summary']['retrieval'] = {
            'recall@3': r.get('recall@3', 0),
            'recall@5': r.get('recall@5', 0),
            'mrr': r.get('mrr', 0),
            'status': '✅' if r.get('recall@3', 0) > 0.5 else '⚠️'
        }

    if 'end2end' in results:
        e = results['end2end']
        report['summary']['end2end'] = {
            'faithfulness': e.get('faithfulness_avg', 0),
            'usefulness': e.get('usefulness_avg', 0),
            'overall': e.get('overall_avg', 0),
            'status': '✅' if e.get('overall_avg', 0) >= 3.5 else '⚠️'
        }

    # 保存报告
    report_path = report_dir / "evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印报告
    print(f"\n{'='*60}")
    print("📋 评估报告")
    print(f"{'='*60}")
    print(f"时间: {report['timestamp']}")

    if 'retrieval' in report['summary']:
        r = report['summary']['retrieval']
        print(f"\n🔍 检索质量:")
        print(f"  Recall@3: {r['recall@3']:.3f} {r['status']}")
        print(f"  Recall@5: {r['recall@5']:.3f}")
        print(f"  MRR:      {r['mrr']:.3f}")

    if 'end2end' in report['summary']:
        e = report['summary']['end2end']
        print(f"\n🎯 端到端质量:")
        print(f"  忠实度: {e['faithfulness']:.1f}/5.0 {e['status']}")
        print(f"  实用性: {e['usefulness']:.1f}/5.0")
        print(f"  综合:   {e['overall']:.1f}/5.0")

    print(f"\n💾 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*60)
    print("🍽️  HowToCook-RAG 分层评估系统")
    print("="*60)

    # 询问要运行哪些评估
    print("\n选择要运行的评估:")
    print("1. 单元测试 (Layer 1)")
    print("2. 检索评估 (Layer 2)")
    print("3. 端到端评估 (Layer 3)")
    print("4. 全部运行")
    print("5. 仅生成报告")

    choice = input("\n输入选择 (1-5): ").strip()

    eval_dir = Path(__file__).parent

    if choice == '1':
        run_command(
            [sys.executable, str(eval_dir / "unit_test.py")],
            "运行单元测试"
        )

    elif choice == '2':
        run_command(
            [sys.executable, str(eval_dir / "retrieval_eval.py")],
            "运行检索评估"
        )

    elif choice == '3':
        run_command(
            [sys.executable, str(eval_dir / "end2end_eval.py")],
            "运行端到端评估"
        )

    elif choice == '4':
        # 运行全部
        success = True
        success &= run_command(
            [sys.executable, str(eval_dir / "unit_test.py")],
            "Layer 1: 单元测试"
        )
        success &= run_command(
            [sys.executable, str(eval_dir / "retrieval_eval.py")],
            "Layer 2: 检索评估"
        )
        success &= run_command(
            [sys.executable, str(eval_dir / "end2end_eval.py")],
            "Layer 3: 端到端评估"
        )

        if success:
            generate_report()

    elif choice == '5':
        generate_report()

    else:
        print("无效选择")


if __name__ == "__main__":
    main()
