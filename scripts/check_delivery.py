"""Check the retained source tree and tracked file-size boundary; no deployment or external reads."""
import re
import subprocess
from pathlib import Path

def main():
    checkout = Path(__file__).resolve().parents[1]
    problems = []
    files = subprocess.check_output(['git', 'ls-files', '-c', '--others', '--exclude-standard', '-z'], cwd=checkout).decode().split('\0')
    for name in filter(None, files):
        path = checkout / name
        catalog = path.resolve().parent == checkout / 'web/src/locales'
        if path.is_file() and path.suffix in ('.py', '.ts', '.tsx', '.mjs', '.js', '.html', '.sh', '.json', '.css') and not catalog:
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r'[\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]', line):
                    problems.append(f'{name}:{number}: CJK outside locale catalog')
                if re.search(r'/(?:Users|private/tmp)/', line):
                    problems.append(f'{name}:{number}: machine-specific source path')
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 10_000_000:
            problems.append(f'{name}: exceeds 10 MB')
    if problems:
        raise SystemExit('\n'.join(problems))
    print('delivery source and file-size checks passed')


if __name__ == '__main__':
    main()
