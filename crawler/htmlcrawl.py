import requests
from lxml import html
import json
import urllib3
import os
import re
import time
import random
from urllib.parse import urljoin

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def crawl_dogye():
    print("=" * 60)
    print("   강원대 도계캠 시설물 고장신고 크롤러")
    print("=" * 60)

    # 도계캠 URL 입력
    target_list_url = input("도계캠 목록 URL을 입력하세요: ").strip()

    if not target_list_url:
        print("[ERROR] URL을 입력해야 합니다.")
        return

    if not target_list_url.startswith(("http://", "https://")):
        target_list_url = "https://" + target_list_url

    # 저장 폴더 고정
    save_folder = r"C:\Users\ch901\PycharmProjects\인공지능project\crawler\data\dogye"
    os.makedirs(save_folder, exist_ok=True)

    print(f"[INFO] 저장 폴더: {save_folder}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
    }

    total_count = 0

    # --------------------------------------
    # 도계캠 페이지 범위는 1~2로 고정
    # --------------------------------------
    for page in range(1, 3):
        sep = "&" if "?" in target_list_url else "?"
        current_list_url = f"{target_list_url}{sep}pageIndex={page}"

        print(f"\n[List] {page}페이지 읽는 중... ({current_list_url})")

        try:
            resp = requests.get(current_list_url, headers=headers, verify=False)
            tree = html.fromstring(resp.content)

            # 게시글 링크
            raw_links = tree.xpath('//a[contains(@href, "nttNo=")]/@href')

            if not raw_links:
                print("   → 이 페이지에는 게시글 없음.")
                continue

            full_links = list({urljoin(current_list_url, link) for link in raw_links})
            print(f"   → {len(full_links)}개 게시글 발견")

            for view_url in full_links:
                match = re.search(r'nttNo=(\d+)', view_url)
                ntt_id = match.group(1) if match else str(int(time.time()))

                process_detail_page(view_url, ntt_id, headers, save_folder)
                total_count += 1

                time.sleep(random.uniform(0.4, 0.8))

        except Exception as e:
            print(f"[ERROR] 페이지 처리 오류: {e}")

    print("=" * 60)
    print(f"[완료] 총 {total_count}개 저장됨 → {save_folder}")


# ---------------------------------------------------------
# 기존 네 코드 그대로 (춘천/삼척/도계 모두 공용)
# ---------------------------------------------------------
def process_detail_page(url, ntt_id, headers, save_folder):
    try:
        resp = requests.get(url, headers=headers, verify=False)
        resp.encoding = 'utf-8'

        if resp.status_code != 200:
            return

        tree = html.fromstring(resp.content)

        # 제목
        title_xpath = '/html/body/div[1]/div[2]/div[2]/main/article/div/div/table[1]/tbody/tr[1]/td'
        title_el = tree.xpath(title_xpath)
        title = title_el[0].text_content().strip() if title_el else "제목없음"

        # 작성자
        poster_xpath = '/html/body/div[1]/div[2]/div[2]/main/article/div/div/table[1]/tbody/tr[2]/td'
        poster_el = tree.xpath(poster_xpath)
        poster = ' '.join(poster_el[0].text_content().split()) if poster_el else ""

        # 내용
        content_xpath = '/html/body/div[1]/div[2]/div[2]/main/article/div/div/table[1]/tbody/tr[3]/td'
        content_el = tree.xpath(content_xpath)

        if content_el:
            target = content_el[0]
            for script in target.xpath(".//script | .//style"):
                script.drop_tree()
            raw_content = target.text_content()
            content = ' '.join(raw_content.split())
        else:
            content = ""

        # 날짜
        timestamp_xpath = '/html/body/div[1]/div[2]/div[2]/main/article/div/div/div[1]/div/span[2]/strong'
        timestamp_el = tree.xpath(timestamp_xpath)
        timestamp = timestamp_el[0].text_content().strip() if timestamp_el else "0000-00-00"

        # 날짜 → 파일 prefix
        nums = re.findall(r'\d+', timestamp)
        date_prefix = f"{nums[0]}{int(nums[1]):02d}{int(nums[2]):02d}" if len(nums) >= 3 else "00000000"

        # 저장
        filename = f"{date_prefix}_{ntt_id}.json"
        filepath = os.path.join(save_folder, filename)

        data = {
            "url": url,
            "id": ntt_id,
            "title": title,
            "poster": poster,
            "timestamp": timestamp,
            "content": content
        }

        with open(filepath, "w", encoding="utf-8-sig") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"[저장] {filename}")

    except Exception as e:
        print(f"[ERROR 상세페이지] {ntt_id}: {e}")


# 실행
if __name__ == "__main__":
    crawl_dogye()
