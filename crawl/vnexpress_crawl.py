from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
import urllib.parse
import time
import csv

import unicodedata

def to_slug(text):
    # Bước 1: chuẩn hóa Unicode (NFD tách dấu khỏi chữ cái)
    text = unicodedata.normalize('NFD', text)
    # Bước 2: loại bỏ dấu (chữ có dấu sẽ bị tách thành 2 phần: base + dấu, ta loại dấu)
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
    # Bước 3: chuyển thành chữ thường và bỏ khoảng trắng
    text = text.lower().replace(' ', '')
    return text

def get_article_data(driver, url):
    driver.get(url)
    time.sleep(4)

    # Lấy title
    try:
        title = driver.find_element(By.CSS_SELECTOR, "h1.title-detail").text.strip()
    except:
        title = ""

    # Lấy ngày đăng
    try:
        date = driver.find_element(By.XPATH, "//span[@class='date']").text.strip()
    except:
        date = ""

    # Mở rộng tất cả các bình luận nếu có nút "Xem thêm bình luận"
    while True:
        try:
            btn = driver.find_element(By.XPATH, "//a[@id='show_more_coment']")
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(1.5)
        except (NoSuchElementException, ElementClickInterceptedException):
            break

    # Lấy danh sách comment
    comments = []
    try:
        comment_elements = driver.find_elements(By.XPATH, "//p[@class='full_content']")
        for cmt in comment_elements:
            text = cmt.text.strip()
            if text:
                comments.append(text)
    except:
        pass

    comment_text = "|||".join(comments)
    return title, date, comment_text

def crawl_vnexpress_articles(keyword, max_pages=5, output_csv="vnexpress_output.csv"):
    output_csv = to_slug(keyword)+"_"+output_csv
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    base_url = "https://timkiem.vnexpress.net/"
    query = urllib.parse.quote(keyword)
    all_links = []

    # Crawl các link từ trang tìm kiếm
    for page in range(1, max_pages + 1):
        search_url = (
            f"{base_url}?q={query}"
            f"&media_type=all&fromdate=0&todate=0&latest=&cate_code="
            f"&search_f=title,tag_list&date_format=all&page={page}"
        )
        driver.get(search_url)
        time.sleep(2)

        article_elements = driver.find_elements(By.XPATH, "//h3[contains(@class, 'title-news')]/a")
        for a in article_elements:
            link = a.get_attribute("href")
            if link and link.startswith("https://"):
                all_links.append(link)

    print(f"✅ Tổng cộng {len(all_links)} bài viết được thu thập")

    # Ghi dữ liệu vào CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Title", "Date", "Comments"])

        for idx, link in enumerate(all_links, 1):
            print(f"🔎 [{idx}/{len(all_links)}] Đang xử lý: {link}")
            try:
                title, date, comments = get_article_data(driver, link)
                writer.writerow([title, date, comments])
            except Exception as e:
                print(f"⚠️ Lỗi với bài viết: {link} – {e}")
                continue

    driver.quit()
    print(f"✅ Dữ liệu đã được lưu vào file: {output_csv}")

# ✅ Gọi hàm chính với từ khóa ví dụ
# crawl_vnexpress_articles("Sơn Tùng", max_pages=5)
crawl_vnexpress_articles("Trấn Thành", max_pages=5)
crawl_vnexpress_articles("Đông Nhi", max_pages=5)
crawl_vnexpress_articles("Sơn Tùng", max_pages=5)
crawl_vnexpress_articles("Hồ Ngọc Hà", max_pages=5)
crawl_vnexpress_articles("Xuân Bắc", max_pages=5)
crawl_vnexpress_articles("Ái Phương", max_pages=5)
crawl_vnexpress_articles("Minh Hằng", max_pages=5)
crawl_vnexpress_articles("Khởi My", max_pages=5)
crawl_vnexpress_articles("Ngô Thanh Vân", max_pages=5)
crawl_vnexpress_articles("Midu", max_pages=5)
# crawl_vnexpress_articles("Sơn Tùng", max_pages=5)