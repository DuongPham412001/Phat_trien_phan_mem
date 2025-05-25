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
    print(f"➡️ Truy cập bài viết: {url}")
    driver.get(url)
    time.sleep(3)

    # Lấy tiêu đề
    try:
        title = driver.find_element(By.CSS_SELECTOR, "h1.article__title.cms-title").text.strip()
        print(f"📝 Tiêu đề: {title}")
    except:
        print("❌ Không lấy được tiêu đề")
        title = ""

    # Lấy ngày đăng
    try:
        date = driver.find_element(By.CSS_SELECTOR, "time.time").get_attribute("datetime").strip()
        print(f"📅 Ngày đăng: {date}")
    except:
        print("❌ Không lấy được ngày đăng")
        date = ""

    return title, date

def crawl_nhandan_articles(keyword, output_csv="nhandan_output.csv"):
    output_csv = to_slug(keyword)+"_"+output_csv
    print(f"🔍 Bắt đầu tìm kiếm với từ khóa: {keyword}")
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    query = urllib.parse.quote(keyword)
    base_url = f"https://nhandan.vn/tim-kiem/?q={query}"
    driver.get(base_url)
    time.sleep(3)

    print("🔁 Đang nhấn 'Xem thêm' để tải thêm bài viết...")
    count_click = 0
    while True:
        try:
            xem_them_btn = driver.find_element(By.XPATH, "//button[normalize-space()='Xem thêm']")
            if not xem_them_btn.is_displayed():
                print("⛔ Nút 'Xem thêm' không còn hiển thị.")
                break

            driver.execute_script("arguments[0].click();", xem_them_btn)
            count_click += 1
            print(f"   ✅ Đã nhấn 'Xem thêm' {count_click} lần")
            time.sleep(2)
        except (NoSuchElementException, ElementClickInterceptedException):
            print("⛔ Không tìm thấy nút 'Xem thêm' nữa.")
            break

    print("📄 Đang lấy danh sách bài viết...")
    article_links = []
    try:
        articles = driver.find_elements(By.XPATH, "//article")
        print(f"🔎 Đã tìm thấy {len(articles)} thẻ <article>")
        for i, article in enumerate(articles, 1):
            try:
                a_tag = article.find_element(By.TAG_NAME, "a")
                link = a_tag.get_attribute("href")
                if link and link.startswith("https://"):
                    article_links.append(link)
                    print(f"   [{i}] ✅ Link: {link}")
            except:
                print(f"   [{i}] ❌ Không tìm được link trong thẻ article")
                continue
    except:
        print("❌ Lỗi khi lấy danh sách bài viết")

    print(f"\n✅ Tổng cộng {len(article_links)} bài viết được thu thập.\n")

    # Ghi vào CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Title", "Date"])

        for idx, link in enumerate(article_links, 1):
            print(f"\n📥 [{idx}/{len(article_links)}] Đang xử lý bài viết...")
            try:
                title, date = get_article_data(driver, link)
                writer.writerow([title, date])
            except Exception as e:
                print(f"⚠️ Lỗi xử lý bài viết: {e}")
                continue

    driver.quit()
    print(f"\n📁 Dữ liệu đã được lưu vào: {output_csv}")
    time.sleep(5)

# ✅ Gọi hàm chính


crawl_nhandan_articles("Trấn Thành")
crawl_nhandan_articles("Đông Nhi")
crawl_nhandan_articles("Sơn Tùng")
crawl_nhandan_articles("Hồ Ngọc Hà")
crawl_nhandan_articles("Xuân Bắc")
crawl_nhandan_articles("Ái Phương")
crawl_nhandan_articles("Minh Hằng")
crawl_nhandan_articles("Khởi My")
crawl_nhandan_articles("Ngô Thanh Vân")
crawl_nhandan_articles("Midu")
# crawl_nhandan_articles("Sơn Tùng")