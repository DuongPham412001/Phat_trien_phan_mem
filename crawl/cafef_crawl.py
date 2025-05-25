assert False 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
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
    print(f"➡️ Truy cập bài viết: {url}")
    driver.get(url)
    time.sleep(2.5)

    # Lấy tiêu đề
    try:
        title = driver.find_element(By.CSS_SELECTOR, "h1.title[data-role='title']").text.strip()
        print(f"📝 Tiêu đề: {title}")
    except:
        print("❌ Không lấy được tiêu đề")
        title = ""

    # Lấy ngày đăng
    try:
        date = driver.find_element(By.CSS_SELECTOR, "span.pdate[data-role='publishdate']").text.strip()
        print(f"📅 Ngày đăng: {date}")
    except:
        print("❌ Không lấy được ngày đăng")
        date = ""

    return title, date

def crawl_cafef_articles(keyword, max_pages=5, output_csv="cafef_output.csv"):
    output_csv = to_slug(keyword)+"_"+output_csv
    print(f"🔍 Bắt đầu tìm kiếm từ khóa: {keyword}")
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    query = urllib.parse.quote(keyword)
    all_links = []

    # Duyệt qua từng trang tìm kiếm
    for page in range(1, max_pages + 1):
        search_url = f"https://cafef.vn/tim-kiem/trang-{page}.chn?keywords={query}"
        print(f"\n📄 Đang truy cập: {search_url}")
        driver.get(search_url)
        time.sleep(2.5)

        try:
            articles = driver.find_elements(By.XPATH, "//div[@class='item']")
            print(f"🔗 Tìm thấy {len(articles)} bài viết ở trang {page}")
            for i, article in enumerate(articles, 1):
                try:
                    a_tag = article.find_element(By.TAG_NAME, "a")
                    link = a_tag.get_attribute("href")
                    if link and link.startswith("https://"):
                        all_links.append(link)
                        print(f"   [{i}] ✅ Link: {link}")
                except:
                    print(f"   [{i}] ❌ Không lấy được link")
        except Exception as e:
            print(f"⚠️ Lỗi khi lấy bài viết ở trang {page}: {e}")

    print(f"\n✅ Tổng cộng {len(all_links)} bài viết được thu thập.\n")

    # Ghi vào file CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Title", "Date"])

        for idx, link in enumerate(all_links, 1):
            print(f"\n📥 [{idx}/{len(all_links)}] Đang xử lý bài viết...")
            try:
                title, date = get_article_data(driver, link)
                writer.writerow([title, date])
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý bài viết: {e}")
                continue

    driver.quit()
    print(f"\n📁 Dữ liệu đã được lưu vào: {output_csv}")
    time.sleep(10)

# ✅ Gọi hàm chính
# crawl_cafef_articles("Sơn Tùng", max_pages=5)
crawl_cafef_articles("Trấn Thành", max_pages=5)
crawl_cafef_articles("Đông Nhi", max_pages=5)
crawl_cafef_articles("Sơn Tùng", max_pages=5)
crawl_cafef_articles("Hồ Ngọc Hà", max_pages=5)
crawl_cafef_articles("Xuân Bắc", max_pages=5)
crawl_cafef_articles("Ái Phương", max_pages=5)
crawl_cafef_articles("Minh Hằng", max_pages=5)
crawl_cafef_articles("Khởi My", max_pages=5)
crawl_cafef_articles("Ngô Thanh Vân", max_pages=5)
crawl_cafef_articles("Midu", max_pages=5)
# crawl_cafef_articles("Sơn Tùng", max_pages=5)