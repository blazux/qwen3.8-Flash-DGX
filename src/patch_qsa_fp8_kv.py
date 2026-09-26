#!/usr/bin/env python3
"""Autorise un cache KV en fp8-e4m3 sur le QSA de Qwen3.8-Flash-Next (vLLM/NVIDIA).

CE QUE CE PATCH FAIT, ET CE QU'IL NE FAIT PAS
---------------------------------------------
vLLM a DEJA pose toute la plomberie de quantisation du cache pour cette couche :
`get_kv_cache_spec` transmet `kv_quant_mode`, `set_default_quant_scales` enregistre
`_k_scale`/`_v_scale`, l'allocation et l'ECRITURE passent par le chemin generique
du cœur. Le seul trou est la LECTURE : les noyaux Triton chargent le cache en
supposant du bf16, et les gardes refusent tout le reste plutot que de lire de
travers.

Ce patch ajoute la dequantisation a la lecture et leve les gardes. Il est INERTE
tant que `--kv-cache-dtype` vaut `auto`/`bfloat16` : `KV_QUANT_MODE` est alors 0,
la branche de dequantisation est eliminee a la compilation Triton, et l'image se
comporte EXACTEMENT comme l'amont.

DEUX BASES : preview (qwen3_8_flash_next) et releases v0.29/v0.30 (qwen4_exp)
-----------------------------------------------------------------------------
Le module s'appelait `qwen3_8_flash_next` dans l'image preview et a ete renomme
`qwen4_exp` dans les releases vLLM >= 0.29. Les deux emplacements sont tries, le
premier trouve gagne. Sur la base preview, chaque ancre est vue exactement une
fois et le script se comporte comme la version qui a construit v0916/v0925
(les paires avant/apres du chemin preview sont verbatim identiques — prouve par
extraction AST, voir docs). Sur v0.30, la plupart des ancres sont identiques ;
celles dont la forme a change ont une variante et le script exige qu'exactement
UNE des deux formes soit vue — jamais zero, jamais les deux — donc aucune
substitution silencieuse n'est possible.

Ce qui change sur v0.30 (outre les noms) :
 - le noyau MQA du selecteur de blocs a quitte ops/qsa.py pour
   ops/qsa_indexer.py et lit deja le dtype compresse nativement ; son cache est
   pilote par `indexer_kv_dtype`, plus par `--kv-cache-dtype` : les points 3/4/9/19
   du chemin preview n'ont plus de cible et sont sautes.
 - le noyau decode est passe en split-K : `TOPK`/`NUM_TILES`/`NUM_SPLITS` sont des
   constexpr, la largeur est `selection_width`, et le bloc N est choisi dans
   `_select_config` (ici 32/64, deja plus petits que sur la preview). La reduction
   du bloc sous quantification se fait donc apres l'appel, avec `num_tiles`
   recalcule.
 - le JIT-warmup (`warmup_qsa_sparse_paged_attention`) lance le meme noyau et
   doit recevoir les nouveaux arguments, sinon la signature ne correspond plus.
   Il doit aussi compiler la specialisation du MODE REEL (point 6c) : figee a 0,
   la variante fp8 ne se compilerait qu'au premier vrai lancement, qui peut se
   faire sous capture du graphe CUDA — ou la compilation Triton est fatale.

POURQUOI REMONTER EN BF16 ET NON EN FP32
-----------------------------------------
MiaAI-Lab, qui a fait le meme travail cote SGLang, remonte K/V en fp32. Remonter
au dtype de la requete (bf16) coute deux fois moins de registres et c'est le
choix d'UPSTREAM VLLM lui-meme (`_cast_kv_tile` : `return (data.to(tl.float32) *
tl.load(tensor_scale)).to(Q.dtype)`). Suivre vLLM ici, c'est heriter de ses
corrections plutot que d'en diverger.

ECHELLES : le wrapper accepte `k_scale`/`v_scale` EN OPTION ; sans eux, la scale
vaut 1.0 et le cache est lu tel quel — exactement le comportement verifie des
images preview (l'owner ne les transmet pas, `set_default_quant_scales`
enregistre 1.0, le checkpoint NVFP4 n'en publie pas). Un checkpoint qui publie
une autre echelle demanderait a cabler `layer._k_scale` dans l'owner ; ce n'est
le cas d'aucun checkpoint teste ici.

NVFP4 n'est pas traite ici : il ajoute un format packe et des echelles fp8
imbriquees. Le fp8-e4m3 valide d'abord la chaine complete.

VERIFICATION -- « ca demarre » ne prouve RIEN
---------------------------------------------
Le mode de defaillance a craindre est SILENCIEUX : le bug de `block_size` du
prefix caching, corrige le 2026-08-30, rendait des reponses plausibles et fausses
sans lever d'erreur. Utiliser `validation/capture_baseline.py` AVANT et
`validation/compare.py` APRES. Le test qui compte est l'egalite des JETONS, pas
l'absence de crash.

Usage :
    python3 patch_qsa_fp8_kv.py <site-packages>
"""
import ast
import os
import sys

SP = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: patch_qsa_fp8_kv.py <site-packages>")

# Nom du module : qwen3_8_flash_next sur la preview, qwen4_exp sur les releases
# vLLM >= 0.29 (renommage amont). Les deux bases ont la meme arborescence.
_CANDIDATES = ("qwen3_8_flash_next", "qwen4_exp")
MODULE = next((m for m in _CANDIDATES if os.path.isdir(f"{SP}/vllm/models/{m}/nvidia")), None)
if MODULE is None:
    sys.exit(f"patch_qsa_fp8_kv: ni {' ni '.join(_CANDIDATES)} sous {SP}/vllm/models/")
BASE = f"{SP}/vllm/models/{MODULE}/nvidia"
OPS = f"{BASE}/ops/qsa.py"
OWNER = f"{BASE}/qsa.py"
IS_V030 = MODULE == "qwen4_exp"


def remplacer(texte: str, avant: str, apres: str, quoi: str) -> str:
    n = texte.count(avant)
    assert n == 1, f"ancre '{quoi}' vue {n} fois (attendu 1) -- l'amont a bouge"
    return texte.replace(avant, apres)


def remplacer_au_choix(texte: str, avant: str, apres: str, quoi: str,
                       avant_v030: str, apres_v030: str, quoi2: str) -> str:
    """`avant` (forme preview) ou `avant_v030` (forme v0.29+) doit apparaitre
    exactement une fois, jamais les deux — donc aucune substitution
    silencieuse n'est possible. Chaque base a SON texte de remplacement :
    les deux formes ne sont pas toujours interchangeables (lancement et
    signature du wrapper different vraiment)."""
    n1, n2 = texte.count(avant), texte.count(avant_v030)
    if n1 == 1 and n2 == 0:
        return texte.replace(avant, apres)
    if n1 == 0 and n2 == 1:
        return texte.replace(avant_v030, apres_v030)
    raise AssertionError(
        f"ancre '{quoi}'/'{quoi2}' vues ({n1}, {n2}) -- attendu (1,0) ou (0,1), "
        "l'amont a bouge"
    )


print(f"patch_qsa_fp8_kv: module {MODULE} ({'release v0.29+' if IS_V030 else 'preview'})")

# ---------------------------------------------------------------- ops/qsa.py
ops = open(OPS).read()

# 0. REUTILISER la dequantisation canonique de vLLM plutot que d'en ecrire une
#    seconde. `_cast_kv_tile` (v1/attention/ops/triton_unified_attention.py) fait
#    deja exactement ce travail pour l'attention unifiee, gere les quatre modes
#    (NONE, FP8 per-tensor, INT8/FP8 per-token-head) ET le cas ou la requete est
#    elle-meme en fp8. Dupliquer la formule aurait cree une seconde verite qui
#    derive : c'est precisement le defaut que ce depot traque partout ailleurs.
ops = remplacer(ops,
    "from vllm.triton_utils import HAS_TRITON, tl, triton",
    "from vllm.triton_utils import HAS_TRITON, tl, triton\n"
    "from vllm.v1.attention.ops.triton_unified_attention import _cast_kv_tile",
    "import du helper canonique")

# 1. Noyau DECODE : signature. Les echelles sont des pointeurs (tenseurs 0-dim
#    cote Python) ; `KV_QUANT_MODE` est une constexpr, donc la branche disparait
#    a la compilation quand elle est fausse. L'ancre est identique sur les deux
#    bases (sur v0.30, PAGE_SIZE suit TOPK, ce qui ne change rien au motif).
ops = remplacer(ops,
    "    num_requests,\n    TOPK: tl.constexpr,",
    "    num_requests,\n"
    "    k_scale_ptr,\n"
    "    v_scale_ptr,\n"
    "    KV_QUANT_MODE: tl.constexpr,\n"
    "    TOPK: tl.constexpr,",
    "signature du noyau decode")

# 2. Noyau DECODE : dequantisation, entre le chargement et le produit scalaire.
#    `keys` est charge tel qu'il est stocke ; s'il est fp8, il vaut
#    quantise * echelle. On remonte, on met a l'echelle, on redescend au dtype de
#    la requete pour que `tl.dot` ait deux operandes de meme type.
ops = remplacer(ops,
    "        scores = tl.dot(query, keys)\n"
    "        # Scaling scores avoids re-quantizing a scaled query to BF16.",
    "        keys = _cast_kv_tile(keys, query, k_scale_ptr, KV_QUANT_MODE)\n"
    "        values = _cast_kv_tile(values, query, v_scale_ptr, KV_QUANT_MODE)\n"
    "        scores = tl.dot(query, keys)\n"
    "        # Scaling scores avoids re-quantizing a scaled query to BF16.",
    "dequantisation du noyau decode")

if not IS_V030:
    # 3. Noyau MQA (cache COMPRESSE, distinct du KV principal) : signature.
    #    Sur v0.30 ce noyau vit dans ops/qsa_indexer.py et lit deja son dtype
    #    compresse nativement : rien a y porter (voir le docstring).
    ops = remplacer(ops,
        "    score_divisor,\n    PAGE_SIZE: tl.constexpr,",
        "    score_divisor,\n"
        "    kc_scale_ptr,\n"
        "    KC_QUANT_MODE: tl.constexpr,\n"
        "    PAGE_SIZE: tl.constexpr,",
        "signature du noyau mqa")

    # 4. Noyau MQA : dequantisation. Il ne lit que les cles (c'est le selecteur de
    #    blocs), pas les valeurs.
    ops = remplacer(ops,
        "        scores = tl.dot(keys, query, out_dtype=tl.float32)",
        "        keys = _cast_kv_tile(keys, query, kc_scale_ptr, KC_QUANT_MODE)\n"
        "        scores = tl.dot(keys, query, out_dtype=tl.float32)",
        "dequantisation du noyau mqa")

# 5. Le garde du wrapper decode. Il exigeait l'egalite des trois dtypes ; on
#    autorise un CACHE fp8 avec une REQUETE bf16, ce qui est precisement le point.
ops = remplacer(ops,
    "    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16",
    "    assert q.dtype == torch.bfloat16\n"
    "    assert k_cache.dtype == v_cache.dtype\n"
    "    _fp8_kv = k_cache.dtype == torch.float8_e4m3fn  # e5m2: the mqa launch has no quant mode for it\n"
    "    # 1 = FP8_PER_TENSOR in KVQuantMode. The per-token-head modes (2, 3)\n"
    "    # apply their scales elsewhere in the loop and are not wired here.\n"
    "    _kv_mode = 1 if _fp8_kv else 0\n"
    "    assert k_cache.dtype == torch.bfloat16 or _fp8_kv, (\n"
    '        f"QSA: KV cache is {k_cache.dtype}, expected bf16 or fp8_e4m3"\n'
    "    )",
    "garde de dtype du wrapper decode")

# 6. Passage des echelles au lancement du noyau decode. Absentes, on retombe sur
#    des echelles neutres : le patch reste alors un no-op numerique.
ops = remplacer_au_choix(ops,
    "        block_table.shape[0],\n        TOPK=logical_indices.shape[1],",
    "        block_table.shape[0],\n"
    "        _k_scale_t,\n"
    "        _v_scale_t,\n"
    "        KV_QUANT_MODE=_kv_mode,\n"
    "        TOPK=logical_indices.shape[1],",
    "lancement du noyau decode",
    "        block_table.shape[0],\n        TOPK=selection_width,",
    # v0.30 : TOPK reste selection_width (logical_indices.shape[1] serait
    # selection_width+1, la colonne de comptage comprise — silencieusement faux).
    "        block_table.shape[0],\n"
    "        _k_scale_t,\n"
    "        _v_scale_t,\n"
    "        KV_QUANT_MODE=_kv_mode,\n"
    "        TOPK=selection_width,",
    "lancement du noyau decode (v0.30, split-K)")

if IS_V030:
    # 6b. Le JIT-warmup lance le meme noyau : il doit recevoir les deux pointeurs
    #     d'echelle et la constexpr, sinon la signature ne correspond plus et le
    #     warmup du boot explose. L'echelle est factice (le warmup ne lance rien,
    #     il compile) ; le MODE, lui, doit etre reel — voir 6c.
    ops = remplacer(ops,
        "    warmed = []",
        "    _warmup_scale = TritonWarmupTensor(torch.float32)\n"
        "    warmed = []",
        "echelle factice du warmup (v0.30)")

    # 6c. LE MODE REEL AU WARMUP (Vorbehalt du review Hyperion-II).
    #
    # Le warmup recoit le VRAI cache (`owner.kv_cache`). Sous fp8, le cœur
    # l allooue en uint8 et le runtime le reinterprete en fp8 juste avant le
    # noyau ; le dtype du pointeur entre dans la specialisation Triton. Fixe a
    # 0 avec un cache uint8, le warmup ne compilait donc AUCUNE des deux
    # specialisations que le premier vrai lancement reclame : la compilation
    # JIT tomberait sur ce premier lancement — et si c'est une capture de graphe
    # CUDA, le boot meurt (pas de compilation sous capture). On reprend ici la
    # REGLE EXACTE du wrapper decode : uint8 (ou fp8) -> mode 1 et reinterpret,
    # bf16 -> mode 0 sans vue, donc le cas bf16 reste identique a l'amont.
    ops = remplacer(ops,
        "    head_dim = kv_cache.shape[-1] // 2\n"
        "    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)",
        "    head_dim = kv_cache.shape[-1] // 2\n"
        "    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)\n"
        "    # Same rule as the decode wrapper: uint8 holds the raw bytes of an\n"
        "    # fp8 cache (the core allocates it that way) -> reinterpret it, and\n"
        "    # derive the constexpr from the storage so the warmup compiles the\n"
        "    # mode the first real launch will actually use.\n"
        "    _warmup_kv_mode = 1 if kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn) else 0\n"
        "    if kv_cache.dtype == torch.uint8:\n"
        "        key_cache = key_cache.view(torch.float8_e4m3fn)\n"
        "        value_cache = value_cache.view(torch.float8_e4m3fn)",
        "mode reel du warmup (v0.30)")
    ops = remplacer(ops,
        "            num_requests,\n            TOPK=selection_width,",
        "            num_requests,\n"
        "            _warmup_scale,\n"
        "            _warmup_scale,\n"
        "            KV_QUANT_MODE=_warmup_kv_mode,\n"
        "            TOPK=selection_width,",
        "lancement du warmup decode (v0.30)")

# 7. Les tenseurs d'echelle, materialises juste avant le lancement.
ops = remplacer(ops,
    "    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](",
    "    _one = torch.ones((), dtype=torch.float32, device=q.device)\n"
    "    _k_scale_t = k_scale if k_scale is not None else _one\n"
    "    _v_scale_t = v_scale if v_scale is not None else _one\n"
    "    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](",
    "materialisation des echelles decode")

# 8. Signature publique du wrapper decode. Les echelles sont OPTIONNELLES : sans
#    elles, `_k_scale_t` retombe sur 1.0 et le patch est un no-op numerique, ce
#    qui garantit que tout appelant non modifie continue de fonctionner.
ops = remplacer_au_choix(ops,
    "    token_to_req: torch.Tensor,\n"
    "    out: torch.Tensor | None = None,\n"
    ") -> torch.Tensor:",
    "    token_to_req: torch.Tensor,\n"
    "    out: torch.Tensor | None = None,\n"
    "    k_scale: torch.Tensor | None = None,\n"
    "    v_scale: torch.Tensor | None = None,\n"
    ") -> torch.Tensor:",
    "signature du wrapper decode",
    "    out: torch.Tensor | None = None,\n"
    "    *,\n"
    "    output_gate: torch.Tensor,\n"
    ") -> torch.Tensor:",
    # v0.30 : gate keyword-only. Les echelles suivent `out`, avant le `*`.
    "    out: torch.Tensor | None = None,\n"
    "    k_scale: torch.Tensor | None = None,\n"
    "    v_scale: torch.Tensor | None = None,\n"
    "    *,\n"
    "    output_gate: torch.Tensor,\n"
    ") -> torch.Tensor:",
    "signature du wrapper decode (v0.30, gate keyword-only)")

if not IS_V030:
    # 19. LE CACHE DU SELECTEUR DE BLOCS, reinterpretation manquante (preview).
    #
    # `qsa_mqa_paged` lit le cache de l'INDEXEUR (`self.indexer.raw_key_cache`),
    # pas le cache KV principal. Sur la preview ce cache suit aussi
    # `kv_cache_dtype` et arrive donc en `uint8` non reinterprete : le noyau lit
    # des OCTETS comme des flottants, la selection des blocs devient aberrante et
    # la sortie diverge des le PREMIER jeton (mesure du 2026-08-30 : 11 sorties
    # sur 12 fausses) alors meme que le chemin de decode est correct.
    #
    # Sur v0.30 : sans cible (voir le docstring) — le selecteur a son propre
    # cache, dtype pilote par `indexer_kv_dtype`, jamais uint8.
    ops = remplacer(ops,
        "    _validate_mqa(q)",
        "    # The block selector reads the indexer cache, a SEPARATE tensor from the\n"
        "    # main KV cache that follows the same dtype. Left as uint8 the kernel\n"
        "    # would read integers and pick arbitrary blocks.\n"
        "    if k_cache.dtype == torch.uint8:\n"
        "        k_cache = k_cache.view(torch.float8_e4m3fn)\n"
        "    _validate_mqa(q)",
        "reinterpretation du cache de l'indexeur")

    # 9. Lancement du noyau mqa : echelle neutre tant que le cache compresse reste bf16.
    ops = remplacer(ops,
        "        float(score_divisor),\n        PAGE_SIZE=k_cache.shape[1],",
        "        float(score_divisor),\n"
        "        torch.ones((), dtype=torch.float32, device=q.device),\n"
        "        1 if k_cache.dtype == torch.float8_e4m3fn else 0,\n"
        "        PAGE_SIZE=k_cache.shape[1],",
        "lancement du noyau mqa")

    # 18. LA MEMOIRE PARTAGEE : reduire le bloc quand le cache est quantifie.
    #
    # `_cast_kv_tile` materialise une tuile fp32 (`data.to(tl.float32)`) avant de
    # redescendre au dtype de la requete. Cette tuile est DEUX FOIS plus large
    # que le bf16 d'origine, sur K et sur V. Mesure au boot sur la preview :
    #
    #     triton.runtime.errors.OutOfResources: out of resource: shared memory,
    #     Required: 106496, Hardware limit: 101376
    #
    # Le GB10 plafonne a 99 KiB de memoire partagee -- la meme limite que le
    # patch FLA amont encode deja (`DEFAULT = 101376  # spark-fla-shmem`). On
    # halve le bloc N quand le cache est quantifie.
    ops = remplacer(ops,
        "    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)",
        "    if _kv_mode != 0:\n"
        "        # _cast_kv_tile materialises an fp32 tile, doubling shared memory for\n"
        "        # K and V. sm_121 caps at 101376 bytes and the kernel asked for\n"
        "        # 106496; halving the N block fits.\n"
        "        block_n = max(16, block_n // 2)\n"
        "    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)",
        "bloc N reduit sous quantisation")
else:
    # 18b. Meme correction sur v0.30, mais le bloc N vient de `_select_config`
    #      (32/64) et `num_tiles` est calcule dans l'appel : on halve apres
    #      l'appel et on recalcule `num_tiles`. Halver ne fait qu'augmenter le
    #      nombre de tuiles, donc `num_splits <= num_tiles` reste vrai.
    ops = remplacer(ops,
        "    block_n, partial_warps, num_tiles, num_splits = _select_config(\n"
        "        q.shape[0], k_cache.shape[2], use_prefill_config, selection_width\n"
        "    )",
        "    block_n, partial_warps, num_tiles, num_splits = _select_config(\n"
        "        q.shape[0], k_cache.shape[2], use_prefill_config, selection_width\n"
        "    )\n"
        "    if _kv_mode != 0:\n"
        "        # _cast_kv_tile materialises an fp32 tile, doubling shared memory for\n"
        "        # K and V. sm_121 caps at 101376 bytes; halving the N block fits\n"
        "        # (same correction as on the preview base, applied post-select).\n"
        "        block_n = max(16, block_n // 2)\n"
        "        num_tiles = triton.cdiv(selection_width, block_n)",
        "bloc N reduit sous quantisation (v0.30)")

ast.parse(ops)
open(OPS, "w").write(ops)
print("ops/qsa.py : noyau decode dequantifiant, gardes de dtype leves"
      + ("" if IS_V030 else " + mqa"))

# -------------------------------------------------------------------- qsa.py
owner = open(OWNER).read()

# 10. Ce que le backend DECLARE savoir faire.
owner = remplacer(owner,
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]',
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
    '        "auto",\n'
    '        "bfloat16",\n'
    '        "fp8",\n'
    '        "fp8_e4m3",\n'
    '    ]',
    "dtypes declares supportes")

# 11..13. Les gardes DECLARATIFS. Ils ne protegeaient rien une fois les noyaux
#         capables ; le seul garde utile etait celui du wrapper (n.5). Sur les
#         releases le message est sur une ligne, sur la preview sur trois.
owner = remplacer_au_choix(owner,
    '        if self.kv_cache_dtype not in ("auto", "bfloat16"):\n'
    "            raise NotImplementedError(\n"
    '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
    "            )",
    '        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):\n'
    "            raise NotImplementedError(\n"
    '                f"Qwen3.8-Flash-Next QSA: {self.kv_cache_dtype} is not supported "\n'
    '                "(bf16 and fp8_e4m3 are)"\n'
    "            )",
    "garde d'init",
    '        if self.kv_cache_dtype not in ("auto", "bfloat16"):\n'
    '            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")',
    '        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):\n'
    '            raise NotImplementedError("QSA: fp8_e4m3 is supported (patched)")',
    "garde d'init (v0.30)")

owner = remplacer_au_choix(owner,
    "        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:\n"
    '            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 Q/K/V")',
    "        if query.dtype != torch.bfloat16:\n"
    '            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires a BF16 query")\n'
    "        if key_cache.dtype not in (\n"
    "            torch.bfloat16,\n"
    "            torch.float8_e4m3fn,\n"
    "            torch.uint8,\n"
    "        ):\n"
    "            raise NotImplementedError(\n"
    '                f"Qwen3.8-Flash-Next QSA: cache dtype {key_cache.dtype} is not supported"\n'
    "            )",
    "garde d'entree du noyau",
    "        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:\n"
    '            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")',
    "        if query.dtype != torch.bfloat16:\n"
    '            raise NotImplementedError("QSA requires a BF16 query")\n'
    "        if key_cache.dtype not in (\n"
    "            torch.bfloat16,\n"
    "            torch.float8_e4m3fn,\n"
    "            torch.uint8,\n"
    "        ):\n"
    "            raise NotImplementedError(\n"
    '                f"QSA: cache dtype {key_cache.dtype} is not supported"\n'
    "            )",
    "garde d'entree du noyau (v0.30)")

owner = remplacer_au_choix(owner,
    "        if self.kv_cache_torch_dtype != torch.bfloat16:\n"
    "            raise NotImplementedError(\n"
    '                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"\n'
    "            )",
    "        if self.kv_cache_torch_dtype not in (\n"
    "            torch.bfloat16,\n"
    "            torch.float8_e4m3fn,\n"
    "            # vLLM ALLOCATES the quantised cache as uint8: raw bytes,\n"
    "            # reinterpreted as fp8 right before the kernel (see the\n"
    "            # `.view()` in `forward_qsa`). Rejecting uint8 rejected the\n"
    "            # only storage the core produces.\n"
    "            torch.uint8,\n"
    "        ):\n"
    "            raise NotImplementedError(\n"
    '                f"Qwen3.8-Flash-Next QSA: storage dtype {self.kv_cache_torch_dtype} "\n'
    '                "is not supported"\n'
    "            )",
    "garde de stockage",
    '        if self.kv_cache_torch_dtype != torch.bfloat16:\n'
    '            raise NotImplementedError("Qwen4Exp QSA requires BF16 cache storage")',
    "        if self.kv_cache_torch_dtype not in (\n"
    "            torch.bfloat16,\n"
    "            torch.float8_e4m3fn,\n"
    "            torch.uint8,\n"
    "        ):\n"
    "            raise NotImplementedError(\n"
    '                f"QSA: storage dtype {self.kv_cache_torch_dtype} "\n'
    '                "is not supported"\n'
    "            )",
    "garde de stockage (v0.30)")

# 15. LE GARDE RATE AU PREMIER PASSAGE, et qui a fait echouer le boot.
#
# Deux gardes vivent dans une AUTRE classe (l'attention, pas l'impl) et testent
# `cache_config.cache_dtype` la ou ceux d'en haut testent `self.kv_cache_dtype` :
# meme intention, texte different. Chercher les gardes par leur MESSAGE ne
# suffit pas, il faut les chercher par ce qu'ils TESTENT.
owner = remplacer_au_choix(owner,
    '        if cache_config.cache_dtype not in ("auto", "bfloat16"):\n'
    "            raise NotImplementedError(\n"
    '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
    "            )",
    '        if cache_config.cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):\n'
    "            raise NotImplementedError(\n"
    '                f"Qwen3.8-Flash-Next QSA: cache_dtype {cache_config.cache_dtype} "\n'
    '                "is not supported (bf16 and fp8_e4m3 are)"\n'
    "            )",
    "garde cache_config.cache_dtype (classe QSAAttention)",
    '        if cache_config.cache_dtype not in ("auto", "bfloat16"):\n'
    '            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")',
    '        if cache_config.cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):\n'
    '            raise NotImplementedError("QSA: fp8_e4m3 is supported (patched)")',
    "garde cache_config.cache_dtype (v0.30)")

# Le garde `quant_config.kv_cache_scheme is not None` (« does not support KV
# quantization ») vise le schema declare DANS LE CHECKPOINT (compressed-tensors),
# pas `--kv-cache-dtype`. Notre chemin ne le declenche pas ; il est laisse en
# place plutot que leve a l'aveugle.

# 17. LE GARDE DU PARENT, qui teste une capacite dont QSA NE DEPEND PAS.
#
# L'impl herite de `FlashAttentionImpl` et appelle `super().__init__()`. Le
# parent refuse un cache quantifie (« FlashAttention does not support ... »)
# parce que SES noyaux ne savent pas le lire sur SM121. Mais QSA ne calcule PAS
# avec les noyaux FlashAttention : il appelle `qsa_sparse_paged_attention`, en
# Triton, et n'herite du parent que pour l'infrastructure. Le garde refuse une
# capacite qui n'est jamais sollicitee.
#
# On neutralise le dtype LE TEMPS de l'init du parent, puis on le restaure.
# `kv_cache_dtype` est le 7e parametre positionnel, d'ou les deux formes.
owner = remplacer(owner,
    "    def __init__(self, *args, **kwargs) -> None:\n"
    "        super().__init__(*args, **kwargs)\n"
    "        if not is_flash_attn_varlen_func_available():",
    "    def __init__(self, *args, **kwargs) -> None:\n"
    "        # FlashAttentionImpl rejects a quantised cache because ITS kernels\n"
    "        # cannot read it on this device. QSA does not use them: it calls\n"
    "        # qsa_sparse_paged_attention (Triton) and only inherits the\n"
    "        # surrounding plumbing. Neutralise the dtype for the parent init,\n"
    "        # then restore it.\n"
    "        _fp8 = (\"fp8\", \"fp8_e4m3\")\n"
    "        _real_kv_dtype = None\n"
    "        if kwargs.get(\"kv_cache_dtype\") in _fp8:\n"
    "            _real_kv_dtype = kwargs[\"kv_cache_dtype\"]\n"
    "            kwargs[\"kv_cache_dtype\"] = \"auto\"\n"
    "        elif len(args) > 6 and args[6] in _fp8:\n"
    "            _real_kv_dtype = args[6]\n"
    "            args = args[:6] + (\"auto\",) + args[7:]\n"
    "        super().__init__(*args, **kwargs)\n"
    "        if _real_kv_dtype is not None:\n"
    "            self.kv_cache_dtype = _real_kv_dtype\n"
    "        if not is_flash_attn_varlen_func_available():",
    "neutralisation du garde herite de FlashAttentionImpl")

# 16. REINTERPRETATION uint8 -> fp8, juste avant le noyau.
#
# Le cache quantifie est ALLOUE en `torch.uint8` par le cœur. Un `tl.load` sur ce
# pointeur rendrait des ENTIERS, et `_cast_kv_tile` convertirait la valeur entiere
# au lieu de decoder le flottant : des nombres plausibles, totalement faux -- le
# mode de defaillance silencieux qu'on traque depuis le debut.
#
# vLLM resout cela par une REINTERPRETATION DE BITS sans copie, comme pour
# l'attention unifiee (`triton_attn.py` : `key_cache.view(self.fp8_dtype)`).
# Preview : apres la canonisation des strides. v0.30 : la decoupe du cache se
# fait par `split` dans `forward_qsa`, sans passe canonicalize — meme placement
# relatif : apres la decoupe, avant le garde d'entree.
if not IS_V030:
    owner = remplacer(owner,
        "        key_cache = canonicalize_singleton_dim_strides(key_cache)\n"
        "        value_cache = canonicalize_singleton_dim_strides(value_cache)",
        "        key_cache = canonicalize_singleton_dim_strides(key_cache)\n"
        "        value_cache = canonicalize_singleton_dim_strides(value_cache)\n"
        "        # uint8 holds the raw bytes of an fp8 cache: REINTERPRET them, do not\n"
        "        # convert (same step triton_attn.py takes for unified attention).\n"
        "        if key_cache.dtype == torch.uint8:\n"
        "            key_cache = key_cache.view(torch.float8_e4m3fn)\n"
        "            value_cache = value_cache.view(torch.float8_e4m3fn)",
        "reinterpretation uint8 -> fp8")
else:
    owner = remplacer(owner,
        "        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)",
        "        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)\n"
        "        # uint8 holds the raw bytes of an fp8 cache: REINTERPRET them, do not\n"
        "        # convert (same step triton_attn.py takes for unified attention).\n"
        "        if key_cache.dtype == torch.uint8:\n"
        "            key_cache = key_cache.view(torch.float8_e4m3fn)\n"
        "            value_cache = value_cache.view(torch.float8_e4m3fn)",
        "reinterpretation uint8 -> fp8 (decoupe par split, v0.30)")

ast.parse(owner)
open(OWNER, "w").write(owner)
print("qsa.py : fp8_e4m3 declare supporte, gardes declaratifs elargis")
print("dequantisation : `_cast_kv_tile` de vLLM (mode 1 = fp8 per-tensor)")
