with open('burn-in-ai/backend/ml/build_master_architecture.py', 'r') as f:
    code = f.read()
code = code.replace("os.path.join(BASE_DIR, '../../../data')", "os.path.join(BASE_DIR, '../../data')")
code = code.replace("os.path.join(BASE_DIR, '../../../results')", "os.path.join(BASE_DIR, '../../results')")
code = code.replace("os.path.join(BASE_DIR, '../../../notebooks')", "os.path.join(BASE_DIR, '../../notebooks')")
with open('burn-in-ai/backend/ml/build_master_architecture.py', 'w') as f:
    f.write(code)
