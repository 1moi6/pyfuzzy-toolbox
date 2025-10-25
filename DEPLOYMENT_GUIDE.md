# 🚀 Guia de Deployment - fuzzy-systems

Este guia contém instruções passo-a-passo para fazer o deploy do pacote `fuzzy-systems` no PyPI.

---

## ✅ PRÉ-REQUISITOS COMPLETADOS

Todos os arquivos necessários foram criados:

- [x] `LICENSE` - MIT License
- [x] `pyproject.toml` - Build system moderno
- [x] `MANIFEST.in` - Controle de arquivos incluídos
- [x] `.gitignore` - Arquivos ignorados pelo Git
- [x] `CHANGELOG.md` - Histórico de versões
- [x] `setup.py` - Configuração atualizada (versão 1.0.0)
- [x] `fuzzy_systems/__init__.py` - Versão consistente (1.0.0)
- [x] `README.md` - Atualizado com badges e instruções de instalação

---

## 📋 CHECKLIST PRÉ-DEPLOY

### 1. Verificar Contas e Credenciais

- [ ] Criar conta no PyPI: https://pypi.org/account/register/
- [ ] Criar conta no TestPyPI: https://test.pypi.org/account/register/
- [ ] Configurar 2FA (Two-Factor Authentication)
- [ ] Gerar API Token no PyPI:
  - Account Settings → API tokens → "Add API token"
  - Scope: "Entire account" (primeira vez) ou "Project: fuzzy-systems" (depois)
  - Copiar e salvar o token (começa com `pypi-`)

### 2. Instalar Ferramentas de Build

```bash
cd /Users/1moi6/Desktop/Minicurso\ Fuzzy/fuzzy_systems

# Atualizar pip
python3 -m pip install --upgrade pip

# Instalar ferramentas de build
pip install --upgrade build twine setuptools wheel
```

### 3. Configurar URLs do GitHub (IMPORTANTE!)

⚠️ **ANTES DE FAZER DEPLOY**, atualizar as URLs nos seguintes arquivos:

**`setup.py`** (linhas 61, 88-92):
```python
url='https://github.com/SEU_USERNAME/fuzzy-systems',
# ...
project_urls={
    'Homepage': 'https://github.com/SEU_USERNAME/fuzzy-systems',
    'Bug Tracker': 'https://github.com/SEU_USERNAME/fuzzy-systems/issues',
    'Source Code': 'https://github.com/SEU_USERNAME/fuzzy-systems',
    # ...
}
```

**`pyproject.toml`** (linhas 78-83):
```toml
[project.urls]
Homepage = "https://github.com/SEU_USERNAME/fuzzy-systems"
Repository = "https://github.com/SEU_USERNAME/fuzzy-systems"
"Bug Tracker" = "https://github.com/SEU_USERNAME/fuzzy-systems/issues"
# ...
```

---

## 🏗️ PASSO 1: BUILD DO PACOTE

### 1.1 Limpar Builds Anteriores

```bash
cd /Users/1moi6/Desktop/Minicurso\ Fuzzy/fuzzy_systems

# Limpar artefatos antigos
rm -rf build/ dist/ *.egg-info/
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
```

### 1.2 Build

```bash
# Build do pacote (cria source e wheel distributions)
python3 -m build
```

**Saída esperada:**
```
Successfully built fuzzy_systems-1.0.0.tar.gz and fuzzy_systems-1.0.0-py3-none-any.whl
```

**Arquivos criados em `dist/`:**
- `fuzzy-systems-1.0.0.tar.gz` - Source distribution
- `fuzzy_systems-1.0.0-py3-none-any.whl` - Wheel distribution

### 1.3 Verificar o Pacote

```bash
# Verificar integridade
twine check dist/*
```

**Saída esperada:**
```
Checking dist/fuzzy-systems-1.0.0.tar.gz: PASSED
Checking dist/fuzzy_systems-1.0.0-py3-none-any.whl: PASSED
```

---

## 🧪 PASSO 2: TESTAR LOCALMENTE

### 2.1 Criar Ambiente Virtual de Teste

```bash
# Criar ambiente limpo
python3 -m venv test_env
source test_env/bin/activate

# Instalar o pacote localmente
pip install dist/fuzzy_systems-1.0.0-py3-none-any.whl
```

### 2.2 Testar Importação

```bash
# Testar import básico
python3 -c "import fuzzy_systems; print(fuzzy_systems.__version__)"
# Deve imprimir: 1.0.0

# Testar criação de sistema
python3 -c "
import fuzzy_systems as fs
system = fs.MamdaniSystem()
print('✓ MamdaniSystem OK')
"
```

### 2.3 Testar Exemplo Completo

```bash
# Rodar um exemplo
python3 examples/01_inference/01_basic_mamdani.py
```

### 2.4 Limpar Ambiente de Teste

```bash
deactivate
rm -rf test_env
```

---

## 🚀 PASSO 3: UPLOAD PARA TESTPYPI

⚠️ **SEMPRE TESTAR NO TESTPYPI PRIMEIRO!**

### 3.1 Upload

```bash
# Upload para TestPyPI
twine upload --repository testpypi dist/*
```

**Credenciais:**
- Username: `__token__`
- Password: `pypi-...` (seu API token do TestPyPI)

**Ou configurar ~/.pypirc:**
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-SEU_TOKEN_PYPI

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-SEU_TOKEN_TESTPYPI
```

### 3.2 Testar Instalação do TestPyPI

```bash
# Criar ambiente limpo
python3 -m venv test_testpypi
source test_testpypi/bin/activate

# Instalar do TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    fuzzy-systems

# Testar
python3 -c "import fuzzy_systems; print(fuzzy_systems.__version__)"

# Limpar
deactivate
rm -rf test_testpypi
```

**Nota:** `--extra-index-url https://pypi.org/simple/` é necessário para instalar dependências (numpy, scipy, etc.) do PyPI real.

### 3.3 Verificar no TestPyPI

Acessar: https://test.pypi.org/project/fuzzy-systems/

Verificar:
- [ ] Versão correta (1.0.0)
- [ ] README renderizando corretamente
- [ ] Links funcionando
- [ ] Classifiers corretos
- [ ] Dependências listadas

---

## 🎯 PASSO 4: UPLOAD PARA PYPI (PRODUÇÃO)

⚠️ **ATENÇÃO:** Não é possível deletar ou sobrescrever versões no PyPI!

### 4.1 Checklist Final

- [ ] Testado no TestPyPI com sucesso
- [ ] Versão correta (1.0.0)
- [ ] URLs do GitHub atualizadas
- [ ] README renderiza perfeitamente
- [ ] Todos os testes passando
- [ ] CHANGELOG.md atualizado
- [ ] Exemplos funcionando

### 4.2 Upload para PyPI

```bash
# Upload para PyPI OFICIAL
twine upload dist/*
```

**Credenciais:**
- Username: `__token__`
- Password: `pypi-...` (seu API token do PyPI)

### 4.3 Verificar Deploy

1. **Aguardar 2-5 minutos** para propagação

2. **Testar instalação:**
```bash
# Ambiente limpo
python3 -m venv test_pypi_prod
source test_pypi_prod/bin/activate

# Instalar
pip install fuzzy-systems

# Testar
python3 -c "import fuzzy_systems; print(fuzzy_systems.__version__)"

# Limpar
deactivate
rm -rf test_pypi_prod
```

3. **Verificar página:** https://pypi.org/project/fuzzy-systems/

---

## 📝 PASSO 5: GIT E GITHUB

### 5.1 Criar Repositório no GitHub

1. Ir para https://github.com/new
2. Nome: `fuzzy-systems`
3. Descrição: "Comprehensive Fuzzy Logic library for Python"
4. Público
5. NÃO inicializar com README (já temos)

### 5.2 Configurar Git Localmente

```bash
cd /Users/1moi6/Desktop/Minicurso\ Fuzzy/fuzzy_systems

# Inicializar git (se ainda não foi)
git init

# Adicionar todos os arquivos
git add .

# Primeiro commit
git commit -m "Initial commit v1.0.0

- Complete Fuzzy Logic library
- Mamdani and Sugeno inference systems
- ANFIS and Wang-Mendel learning
- Fuzzy ODE solver
- p-Fuzzy dynamic systems
- 16 comprehensive examples
- Full documentation"

# Adicionar remote
git remote add origin https://github.com/SEU_USERNAME/fuzzy-systems.git

# Push
git branch -M main
git push -u origin main
```

### 5.3 Criar Release no GitHub

```bash
# Criar tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

No GitHub:
1. Ir para "Releases" → "Draft a new release"
2. Choose tag: `v1.0.0`
3. Release title: `fuzzy-systems v1.0.0`
4. Description: Copiar do CHANGELOG.md
5. Anexar `dist/fuzzy-systems-1.0.0.tar.gz`
6. Anexar `dist/fuzzy_systems-1.0.0-py3-none-any.whl`
7. Publish release

---

## 🎉 PASSO 6: PÓS-DEPLOY

### 6.1 Atualizar Links

Agora que o repositório GitHub existe, atualizar badges no README.md:

```markdown
[![GitHub](https://img.shields.io/github/stars/SEU_USERNAME/fuzzy-systems?style=social)](https://github.com/SEU_USERNAME/fuzzy-systems)
[![GitHub issues](https://img.shields.io/github/issues/SEU_USERNAME/fuzzy-systems)](https://github.com/SEU_USERNAME/fuzzy-systems/issues)
```

### 6.2 Verificações Finais

- [ ] `pip install fuzzy-systems` funciona
- [ ] Página do PyPI mostra tudo correto
- [ ] GitHub repository está público
- [ ] Release v1.0.0 criada
- [ ] README renderiza no PyPI
- [ ] Links funcionam

### 6.3 Divulgação (Opcional)

- [ ] Tweet/post sobre o lançamento
- [ ] Post em comunidades Python (Reddit r/Python, etc.)
- [ ] Anunciar em grupos acadêmicos de fuzzy logic

---

## 🔄 RELEASES FUTURAS

### Para Versões Subsequentes

1. **Fazer mudanças no código**

2. **Atualizar versão:**
```python
# fuzzy_systems/__init__.py
__version__ = '1.0.1'  # ou 1.1.0, ou 2.0.0
```

3. **Atualizar CHANGELOG.md:**
```markdown
## [1.0.1] - 2024-XX-XX

### Fixed
- Bug fix description

### Added
- New feature description
```

4. **Commit e tag:**
```bash
git add .
git commit -m "Release v1.0.1"
git tag -a v1.0.1 -m "Release v1.0.1"
git push && git push --tags
```

5. **Build e upload:**
```bash
rm -rf dist/ build/ *.egg-info/
python3 -m build
twine upload dist/*
```

---

## ⚠️ PROBLEMAS COMUNS

### "Package name already taken"
- **Solução**: Escolher nome diferente (ex: `fuzzy-systems-lib`, `pyfuzzy-systems`)

### "Version 1.0.0 already exists"
- **Solução**: Incrementar versão (não é possível sobrescrever)

### "Invalid README"
- **Solução**: Validar markdown, verificar `long_description_content_type='text/markdown'`

### "Import Error after install"
- **Solução**: Verificar que `packages=find_packages()` está correto no setup.py

---

## 📞 SUPORTE

- **Issues**: https://github.com/SEU_USERNAME/fuzzy-systems/issues
- **PyPI**: https://pypi.org/project/fuzzy-systems/
- **TestPyPI**: https://test.pypi.org/project/fuzzy-systems/

---

**Pronto para deploy! 🚀**
