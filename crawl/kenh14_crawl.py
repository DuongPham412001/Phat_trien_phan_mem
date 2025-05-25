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
        title = driver.find_element(By.CSS_SELECTOR, "h1.kbwc-title").text.strip()
        print(f"📝 Tiêu đề: {title}")
    except:
        print("❌ Không lấy được tiêu đề")
        title = ""

    # Lấy ngày đăng
    try:
        date = driver.find_element(By.CSS_SELECTOR, "span.kbwcm-time").get_attribute("title").strip()
        print(f"📅 Ngày đăng: {date}")
    except:
        print("❌ Không lấy được ngày đăng")
        date = ""

    return title, date

def scroll_to_bottom(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)

def crawl_kenh14_articles(keyword, output_csv="kenh14_output.csv", max_clicks=3):
    output_csv = to_slug(keyword)+"_"+output_csv
    print(f"🔍 Tìm kiếm từ khóa: {keyword}")
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    query = urllib.parse.quote(keyword)
    search_url = f"https://kenh14.vn/tim-kiem.chn?keywords={query}"
    driver.get(search_url)
    time.sleep(3)

    # Cuộn xuống đáy nhiều lần và ấn "Bấm để xem thêm" tối đa max_clicks lần
    click_count = 0
    while click_count < max_clicks:
        for i in range(4):
            scroll_to_bottom(driver)
        try:
            more_btn = driver.find_element(By.XPATH, "//a[contains(text(),'Bấm để xem thêm')]")
            if more_btn.is_displayed():
                driver.execute_script("arguments[0].click();", more_btn)
                click_count += 1
                print(f"   🔁 Đã bấm 'Xem thêm' lần {click_count}")
                time.sleep(2.5)
            else:
                break
        except NoSuchElementException:
            print("⛔ Không tìm thấy nút 'Bấm để xem thêm'")
            break

    print("📄 Đang lấy danh sách bài viết...")
    article_links = []
    try:
        articles = driver.find_elements(By.XPATH, "//li[contains(@class, 'knswli')]")
        print(f"🔗 Tìm thấy {len(articles)} thẻ <li>")
        for i, article in enumerate(articles, 1):
            try:
                a_tag = article.find_element(By.TAG_NAME, "a")
                link = a_tag.get_attribute("href")
                if link and link.startswith("https://kenh14.vn"):
                    article_links.append(link)
                    print(f"   [{i}] ✅ Link: {link}")
            except:
                continue
    except:
        print("❌ Không lấy được danh sách bài viết")

    print(f"\n✅ Tổng cộng {len(article_links)} bài viết được thu thập.\n")
    # assert False
    # Ghi vào file CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Title", "Date"])

        for idx, link in enumerate(article_links, 1):
            print(f"\n📥 [{idx}/{len(article_links)}] Đang xử lý bài viết...")
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
# crawl_kenh14_articles("Sơn Tùng")
crawl_kenh14_articles("Trấn Thành", max_clicks=3)
crawl_kenh14_articles("Đông Nhi", max_clicks=3)
crawl_kenh14_articles("Sơn Tùng", max_clicks=3)
crawl_kenh14_articles("Hồ Ngọc Hà", max_clicks=3)
crawl_kenh14_articles("Xuân Bắc", max_clicks=3)
crawl_kenh14_articles("Ái Phương", max_clicks=3)
crawl_kenh14_articles("Minh Hằng", max_clicks=3)
crawl_kenh14_articles("Khởi My", max_clicks=3)
crawl_kenh14_articles("Ngô Thanh Vân", max_clicks=3)
crawl_kenh14_articles("Midu", max_clicks=3)
# crawl_kenh14_articles("Sơn Tùng", max_clicks=3)