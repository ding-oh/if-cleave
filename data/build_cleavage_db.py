#!/usr/bin/env python3
"""
CleavgDB_clean 데이터베이스 생성 통합 파이프라인

이 스크립트는 다음 단계를 통합합니다:
1. PDB 파일 다운로드 (RCSB PDB 또는 AlphaFold DB)
2. PDB 파일 renumbering (잔기 번호 정리)
3. CleavgDB_clean 폴더 구조 생성 (epitope 매칭, info.json 생성)
"""

import os
import glob
import json
import requests
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.SeqUtils import seq1
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import warnings
import argparse

warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: PDB 다운로드 함수들
# ============================================================

import re


def get_best_pdb_from_uniprot(uniprot_id: str, save_dir: str = './uniprot_txt_files') -> Tuple[Optional[str], Optional[str]]:
    """
    UniProt ID로부터 가장 좋은 PDB ID 찾기

    UniProt 텍스트 파일을 다운로드/파싱하여:
    1. 가장 좋은 resolution의 PDB ID 반환
    2. PDB가 없으면 AlphaFold ID 반환

    Returns:
        (pdb_id or alphafold_id, allele_isotype)
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{uniprot_id}.txt")

    # UniProt 텍스트 파일 다운로드 또는 캐시에서 읽기
    if not os.path.exists(file_path):
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(file_path, 'w') as f:
                    f.write(response.text)
                response_text = response.text
            else:
                return None, None
        except Exception as e:
            return None, None
    else:
        with open(file_path, 'r') as f:
            response_text = f.read()

    lines = response_text.split("\n")
    best_pdb_id = None
    best_resolution = float('inf')
    alpha_fold_id = None
    allele_isotype = None

    for line in lines:
        # PDB 정보 파싱 (예: DR   PDB; 1FV1; X-ray; 2.00 A; A/B=1-100.)
        if "DR   PDB;" in line:
            try:
                parts = re.split(r';\s*', line.strip())
                if len(parts) >= 4:
                    pdb_id = parts[1].strip()
                    resolution_str = parts[3].strip()
                    # Resolution 파싱 (예: "2.00 A" -> 2.00)
                    if resolution_str != "-" and "A" in resolution_str:
                        resolution = float(resolution_str.split()[0])
                    else:
                        resolution = float('inf')

                    if resolution < best_resolution:
                        best_resolution = resolution
                        best_pdb_id = pdb_id
            except (ValueError, IndexError):
                continue

        # AlphaFold ID 파싱
        elif "DR   AlphaFoldDB;" in line:
            try:
                alpha_fold_id = line.split(";")[1].strip()
            except IndexError:
                pass

        # Gene name 파싱
        elif line.startswith("GN   "):
            match = re.search(r"Name=([\w-]+);", line)
            if match:
                allele_isotype = match.group(1)

    return (best_pdb_id if best_pdb_id else alpha_fold_id), allele_isotype


def download_pdb_from_rcsb(pdb_id: str, directory: str = "./PDB_downloads") -> Optional[str]:
    """RCSB PDB에서 PDB 파일 다운로드"""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{pdb_id}.pdb")

    if os.path.exists(file_path):
        return file_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200 and not response.text.startswith("<!DOCTYPE"):
            with open(file_path, 'w') as f:
                f.write(response.text)
            return file_path
    except Exception as e:
        print(f"RCSB 다운로드 실패 ({pdb_id}): {e}")
    return None


def download_pdb_from_alphafold(uniprot_id: str, directory: str = "./PDB_downloads") -> Optional[str]:
    """AlphaFold DB에서 PDB 파일 다운로드"""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{uniprot_id}.pdb")

    if os.path.exists(file_path):
        return file_path

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        response = requests.get(api_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                pdb_url = data[0].get('pdbUrl')
            else:
                pdb_url = data.get('pdbUrl')

            if pdb_url:
                pdb_response = requests.get(pdb_url, timeout=30)
                if pdb_response.status_code == 200:
                    with open(file_path, 'w') as f:
                        f.write(pdb_response.text)
                    return file_path
    except Exception as e:
        print(f"AlphaFold 다운로드 실패 ({uniprot_id}): {e}")
    return None


def download_pdb(pdb_id: str, uniprot_id: str = None, directory: str = "./PDB_downloads") -> Optional[str]:
    """
    PDB 다운로드 (우선순위):
    1. CSV에 PDB_ID가 있으면 RCSB에서 다운로드
    2. UniProt ID로 가장 좋은 PDB ID 찾아서 다운로드
    3. PDB가 없으면 AlphaFold에서 다운로드
    """
    # 1. PDB_ID가 이미 있으면 RCSB에서 다운로드
    if pdb_id and len(pdb_id) == 4:
        result = download_pdb_from_rcsb(pdb_id, directory)
        if result:
            return result

    # 2. UniProt ID로 가장 좋은 PDB 찾기
    if uniprot_id:
        best_pdb_id, _ = get_best_pdb_from_uniprot(uniprot_id)

        if best_pdb_id:
            # PDB ID면 RCSB에서 다운로드
            if len(best_pdb_id) == 4:
                result = download_pdb_from_rcsb(best_pdb_id, directory)
                if result:
                    return result

            # AlphaFold ID면 AlphaFold에서 다운로드
            return download_pdb_from_alphafold(best_pdb_id, directory)

        # 3. 위 방법 모두 실패시 직접 AlphaFold에서 시도
        return download_pdb_from_alphafold(uniprot_id, directory)

    return None


# ============================================================
# STEP 2: PDB Renumbering 함수들
# ============================================================

def renumber_pdb_file(file_path: str) -> bool:
    """PDB 파일의 잔기 번호를 1부터 순차적으로 재정렬"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        atom_lines = [line for line in lines if line.startswith("ATOM") and len(line) >= 27]

        # Altloc 제거
        filtered_lines = []
        for line in atom_lines:
            altloc = line[16:17]
            if altloc != ' ':
                continue
            clean_line = line[:16] + ' ' + line[17:20] + line[20:]
            filtered_lines.append(clean_line)

        # 중복 잔기 제거
        seen = {}
        kept_lines = []
        for line in filtered_lines:
            chain = line[21:22].strip()
            full_res = line[22:26].strip()
            numeric_res = ''.join(filter(str.isdigit, full_res))
            key = (chain, numeric_res)
            if key not in seen:
                seen[key] = full_res
                kept_lines.append(line)
            elif seen[key] == full_res:
                kept_lines.append(line)

        # 체인별 잔기 수집
        residues_by_chain = defaultdict(list)
        for line in kept_lines:
            chain = line[21:22].strip()
            resseq = line[22:26].strip()
            icode = line[26:27]
            key = (chain, resseq, icode)
            if key not in residues_by_chain[chain]:
                residues_by_chain[chain].append(key)

        # Renumbering 맵 생성
        renumber_map = {}
        for chain, res_keys in residues_by_chain.items():
            for i, key in enumerate(res_keys):
                renumber_map[key] = str(i + 1).rjust(4)

        # 새 라인 생성
        processed_lines = []
        for line in kept_lines:
            chain = line[21:22].strip()
            resseq = line[22:26].strip()
            icode = line[26:27]
            key = (chain, resseq, icode)
            new_resseq = renumber_map.get(key, resseq.rjust(4))
            new_line = line[:22] + new_resseq + ' ' + line[27:]
            processed_lines.append(new_line)

        with open(file_path, 'w') as f:
            f.writelines(processed_lines)

        return True
    except Exception as e:
        print(f"Renumbering 실패 ({file_path}): {e}")
        return False


# ============================================================
# STEP 3: CleavgDB 생성 함수들
# ============================================================

AA_GROUPS = {
    'nonpolar': set('GAVLIMFWP'),
    'polar_uncharged': set('STNQYC'),
    'polar_positive': set('KRH'),
    'polar_negative': set('DE'),
}


def get_group(aa: str) -> Optional[str]:
    for group, members in AA_GROUPS.items():
        if aa in members:
            return group
    return None


def evaluate_match(epitope: str, segment: str) -> bool:
    """Fuzzy matching으로 epitope 매칭 평가"""
    if abs(len(epitope) - len(segment)) > 1:
        return False
    if epitope[0] != segment[0] or epitope[-1] != segment[-1]:
        return False

    same_class_subs = 0
    diff_class_subs = 0
    deletions = abs(len(epitope) - len(segment))

    for a, b in zip(epitope, segment):
        if a != b:
            if get_group(a) == get_group(b):
                same_class_subs += 1
            else:
                diff_class_subs += 1

    return same_class_subs <= 2 and diff_class_subs <= 1 and deletions <= 1


def find_epitope_locations(sequence: str, epitope_seq: str) -> List[Tuple[int, int]]:
    """시퀀스에서 epitope 위치 찾기"""
    matches = []
    ep_len = len(epitope_seq)

    # Exact match 먼저
    for i in range(len(sequence) - ep_len + 1):
        segment = sequence[i:i + ep_len]
        if segment == epitope_seq:
            matches.append((i, i + ep_len - 1))

    if matches:
        return matches

    # Fuzzy match
    for i in range(len(sequence) - ep_len + 1):
        segment = sequence[i:i + ep_len]
        if evaluate_match(epitope_seq, segment):
            matches.append((i, i + ep_len - 1))

    return matches


class ChainSelect(Select):
    def __init__(self, chain_ids, labeled_residues=None, initialize_bfactor=False):
        self.chain_ids = chain_ids
        self.labeled_residues = labeled_residues if labeled_residues else {}
        self.initialize_bfactor = initialize_bfactor

    def accept_chain(self, chain):
        return chain.id in self.chain_ids

    def accept_atom(self, atom):
        residue = atom.get_parent()
        rid = residue.id
        atom.set_occupancy(1.00)
        if self.initialize_bfactor:
            atom.set_bfactor(0.00)
        elif self.labeled_residues and rid in self.labeled_residues:
            atom.set_bfactor(1.00)
        else:
            atom.set_bfactor(0.00)
        return True


class RegionSelect(Select):
    def __init__(self, chain_id, start, end):
        self.chain_id = chain_id
        self.start = start
        self.end = end

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return self.start <= residue.id[1] <= self.end


def filter_unique_chains(structure) -> List[str]:
    """중복 시퀀스를 가진 체인 필터링"""
    sequences = {}
    unique_ids = []
    for model in structure:
        for chain in model:
            seq = ''.join(seq1(res.resname) for res in chain if res.id[0] == ' ')
            if seq not in sequences:
                sequences[seq] = chain.id
                unique_ids.append(chain.id)
    return unique_ids


def create_windows(start: int, end: int, size: int, total: int):
    """Cleavage window 생성"""
    return (
        (max(0, start - size), max(0, start - 1)),  # n_window: start-1이 음수면 0
        (start + 1, end + 1),  # epitope_window
        (min(end + 1, total - 1), min(end + size, total - 1))  # c_window
    )


def save_region_pdb(io: PDBIO, folder: str, name: str, chain: str, start: int, end: int):
    path = os.path.join(folder, name)
    io.save(path, select=RegionSelect(chain, start, end))


def update_bfactor(filepath: str, target_residues: List[int]):
    """B-factor 업데이트 (모든 원자에 대해)"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("ATOM"):
            try:
                resseq = int(line[22:26])
                if resseq in target_residues:
                    bval = 1.00
                else:
                    bval = 0.00
                b_str = f"{bval:6.2f}"
                line = line[:60] + b_str + line[66:]
            except ValueError:
                pass
        new_lines.append(line)

    with open(filepath, 'w') as f:
        f.writelines(new_lines)


def process_row(args):
    """단일 행 처리 (멀티프로세싱용)"""
    idx, row, output_dir, pdb_dir = args

    try:
        pdb_id = row.get('PDB_ID', '')
        pdb_path = row.get('PDB_Path', '')

        # PDB 파일이 없으면 다운로드 시도
        if not pdb_path or not os.path.exists(pdb_path):
            uniprot_id = row.get('UniProt ID', '')
            pdb_path = download_pdb(pdb_id, uniprot_id, pdb_dir)
            if not pdb_path:
                return None

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_path)
        unique_chains = filter_unique_chains(structure)

        for model in structure:
            for chain in model:
                if chain.id not in unique_chains:
                    continue

                residues = [res for res in chain if res.id[0] == ' ']
                chain_seq = ''.join(seq1(res.resname) for res in residues)
                locations = find_epitope_locations(chain_seq, row['Description'])

                if not locations:
                    continue

                folder = os.path.join(output_dir, f"{pdb_id}_{chain.id}")
                os.makedirs(folder, exist_ok=True)

                protein_path = os.path.join(folder, 'protein.pdb')
                io = PDBIO()
                io.set_structure(chain)

                if not os.path.exists(protein_path):
                    io.save(protein_path, select=ChainSelect([chain.id], initialize_bfactor=True))

                index_to_resid = {i: res.id for i, res in enumerate(residues)}
                bfactor_residues = []

                for ep_no, (start, end) in enumerate(locations, 1):
                    ep_dir = os.path.join(folder, f"epitope{idx}_{ep_no}")
                    os.makedirs(ep_dir, exist_ok=True)
                    windows = create_windows(start, end, 9, len(chain_seq))

                    for pos in [start, end]:
                        res_id = index_to_resid.get(pos)
                        if res_id:
                            bfactor_residues.append(res_id[1])

                    save_region_pdb(io, ep_dir, 'epitope.pdb', chain.id, *windows[1])
                    save_region_pdb(io, ep_dir, 'cleavage_n.pdb', chain.id, *windows[0])
                    save_region_pdb(io, ep_dir, 'cleavage_c.pdb', chain.id, *windows[2])

                    with open(os.path.join(ep_dir, 'info.json'), 'w') as f:
                        json.dump({
                            "PDB ID": pdb_id,
                            "Epitope Chain": chain.id,
                            "Epitope Sequence": row['Description'],
                            "Protein Sequence": row.get('Protein Sequence', ''),
                            "Parent Protein IRI": row.get('Parent Protein IRI', ''),
                            "Method/Technique": row.get('Method/Technique', ''),
                            "Allele Name": row.get('Allele Name', ''),
                            "MHC Family": row.get('MHC Family', ''),
                            "MHC allele class": row.get('MHC allele class', ''),
                            "UniProt ID": row.get('UniProt ID', ''),
                            "n_cleavage": start,
                            "c_cleavage": end
                        }, f, indent=2)

                if bfactor_residues:
                    update_bfactor(protein_path, list(set(bfactor_residues)))

        return True
    except Exception as e:
        return None


def clean_empty_folders(directory: str) -> int:
    """빈 폴더 정리"""
    empty = 0
    for root, dirs, _ in os.walk(directory, topdown=False):
        for d in dirs:
            full = os.path.join(root, d)
            if os.path.isdir(full) and not os.listdir(full):
                os.rmdir(full)
                empty += 1
    return empty


def analyze_bfactor_distribution(base_dir: str):
    """B-factor 분포 분석"""
    total_atoms = labeled_atoms = 0
    for root, _, files in os.walk(base_dir):
        if 'protein.pdb' in files:
            with open(os.path.join(root, 'protein.pdb'), 'r') as f:
                for line in f:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        total_atoms += 1
                        try:
                            b = float(line[60:66])
                            if b == 1.0:
                                labeled_atoms += 1
                        except ValueError:
                            continue

    if total_atoms > 0:
        print("\n=== B-factor Labeling 통계 ===")
        print(f"Total CA atoms: {total_atoms}")
        print(f"Labeled (B=1.0): {labeled_atoms} ({(labeled_atoms/total_atoms)*100:.2f}%)")
        print(f"Unlabeled (B=0.0): {total_atoms - labeled_atoms} ({((total_atoms - labeled_atoms)/total_atoms)*100:.2f}%)")


# ============================================================
# 메인 파이프라인
# ============================================================

def run_pipeline(
    csv_path: str,
    output_dir: str = "./CleavgDB_clean",
    pdb_dir: str = "./PDB_downloads",
    nrows: int = None,
    n_workers: int = None
):
    """
    전체 파이프라인 실행

    Args:
        csv_path: epitope 정보가 담긴 CSV 파일 경로
        output_dir: 출력 디렉토리
        pdb_dir: PDB 다운로드 디렉토리
        nrows: 처리할 행 수 (None이면 전체)
        n_workers: 병렬 처리 워커 수
    """
    print(f"CSV 파일 로드: {csv_path}")
    df = pd.read_csv(csv_path)

    if nrows:
        df = df.head(nrows)
        print(f"   처음 {nrows}개 행만 처리")

    print(f"총 {len(df)}개 행 처리 예정")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pdb_dir, exist_ok=True)

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"워커 수: {n_workers}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"PDB 다운로드 디렉토리: {pdb_dir}")
    print()

    # 멀티프로세싱으로 처리
    tasks = [(i, row.to_dict(), output_dir, pdb_dir) for i, row in df.iterrows()]

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_row, tasks),
            total=len(tasks),
            desc="Processing"
        ))

    success_count = sum(1 for r in results if r is not None)
    print(f"\n처리 완료: {success_count}/{len(tasks)}")

    # 빈 폴더 정리
    empty_count = clean_empty_folders(output_dir)
    print(f"빈 폴더 {empty_count}개 삭제됨")

    # 통계 출력
    analyze_bfactor_distribution(output_dir)

    # 결과 요약
    folder_count = len([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])
    print(f"\n생성된 단백질 폴더: {folder_count}개")
    print(f"위치: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description='CleavgDB_clean 데이터베이스 생성 파이프라인',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python build_cleavage_db.py --csv data.csv --output ./CleavgDB_clean
  python build_cleavage_db.py --csv data.csv --nrows 100 --workers 4
        """
    )

    parser.add_argument('--csv', required=True, help='입력 CSV 파일 경로 (epitope 정보)')
    parser.add_argument('--output', default='./CleavgDB_clean', help='출력 디렉토리')
    parser.add_argument('--pdb-dir', default='./PDB_downloads', help='PDB 다운로드 디렉토리')
    parser.add_argument('--nrows', type=int, default=None, help='처리할 행 수 (기본: 전체)')
    parser.add_argument('--workers', type=int, default=None, help='병렬 워커 수')

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV 파일을 찾을 수 없습니다: {args.csv}")
        return

    run_pipeline(
        csv_path=args.csv,
        output_dir=args.output,
        pdb_dir=args.pdb_dir,
        nrows=args.nrows,
        n_workers=args.workers
    )


if __name__ == '__main__':
    main()
