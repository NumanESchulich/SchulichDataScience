import time
import random
import re
import csv
import requests
from bs4 import BeautifulSoup


# Step 1) Use the requests module to download the HTML for URL.
def check_and_install_package(package_name):
    """Check if the package is installed, and install it if not."""
    try:
        __import__(package_name)
    except ImportError:
        print(f"'{package_name}' is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

check_and_install_package("requests")
check_and_install_package("bs4")

def download_html(url):
    """Download the HTML content of the given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return None


# Step 2) Extract the total number of reviews.
def extract_company_name(html_content):
    """Extract the company name from the HTML title tag."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        title_tag = soup.title
        if title_tag:
            title_text = title_tag.get_text()
            company_name = title_text.split(" Reviews")[0].strip()
            return company_name
        return "Unknown Company"
    except Exception as e:
        print(f"Error extracting company name: {e}")
        return "Unknown Company"

def extract_total_reviews(html_content):
    """Extract the total number of reviews from the page."""
    match = re.search(r'"reviewCount":(\d+)', html_content)
    if match:
        try:
            review_count = int(match.group(1))
            return review_count
        except ValueError as e:
            print(f"Error extracting review count: {e}")
            return None
    print("Could not find 'reviewCount' in the page.")
    return None


# Step 3) Iterate over the review pages.
def find_next_page(soup):
    """Find the URL for the next page of reviews."""
    next_page_element = soup.find('a', {'aria-label': 'Next page'})
    if next_page_element and 'href' in next_page_element.attrs:
        next_page_url = 'https://ca.trustpilot.com' + next_page_element['href']
        return next_page_url
    return None


# Step 4) From each page, extract the reviews.
def extract_reviews_from_page(soup, company_name):
    """Extract reviews from the current page."""
    reviews = []
    review_elements = soup.find_all('article', {'data-service-review-card-paper': True})

    print(f"Found {len(review_elements)} reviews on this page.")

    for review_element in review_elements:
        try:
            date_element = review_element.find('time')
            # Extract only the first 10 characters (YYYY-MM-DD) from the date string
            date_published = date_element['datetime'][:10] if date_element else None

            rating_element = review_element.find('img', alt=re.compile(r'(\d) out of 5 stars'))
            if rating_element:
                rating_value_match = re.search(r'(\d) out of 5 stars', rating_element['alt'])
                rating_value = int(rating_value_match.group(1)) if rating_value_match else None
            else:
                rating_value = None

            review_body_element = review_element.find(
                'p', {'data-service-review-text-typography': True}
            )
            review_body = review_body_element.get_text(strip=True) if review_body_element else None

            review_data = {
                'companyName': company_name,
                'datePublished': date_published,
                'ratingValue': rating_value,
                'reviewBody': review_body
            }
            reviews.append(review_data)
        except Exception as e:
            print(f"Error extracting review: {e}")
            continue

    return reviews


# Step 5) From each review, store to the CSV file: companyName, datePublished, ratingValue, reviewBody.
def iterate_review_pages(start_url):
    """Iterate through all review pages and extract reviews."""
    url = start_url
    page_number = 1
    total_reviews_extracted = 0

    html_content = download_html(url)
    if html_content:
        company_name = extract_company_name(html_content)
        print(f"Company Name: {company_name}")
        
        total_reviews = extract_total_reviews(html_content)
        if total_reviews:
            print(f"Total number of reviews found: {total_reviews}")
        else:
            print("Could not retrieve total number of reviews.")
        
        while url:
            print(f"Processing page {page_number}: {url}")
            if page_number > 1:
                html_content = download_html(url)
                if not html_content:
                    print("Failed to download or parse the page.")
                    break
                soup = BeautifulSoup(html_content, 'html.parser')
            else:
                soup = BeautifulSoup(html_content, 'html.parser')

            reviews = extract_reviews_from_page(soup, company_name)
            total_reviews_extracted += len(reviews)
            yield reviews, company_name

            next_url = find_next_page(soup)
            if next_url:
                url = next_url
                page_number += 1
                time.sleep(random.uniform(2, 5))
            else:
                print("All pages have been processed.")
                break
        
        print(f"Total reviews extracted: {total_reviews_extracted}")
    else:
        print("Failed to download or parse the first page.")


# Step 6) Save the final CSV file with four columns: "companyName", "datePublished", "ratingValue", "reviewBody".
def save_reviews_to_csv(reviews, company_name):
    """Save the extracted reviews to a CSV file."""
    filename = f"{company_name} Trustpilot Reviews.csv"
    fieldnames = ['companyName', 'datePublished', 'ratingValue', 'reviewBody']
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)

    print(f"Saved {len(reviews)} reviews to {filename}")


if __name__ == "__main__":
    all_reviews = []
    company_name = "Unknown Company"  # Default value in case the extraction fails

    for reviews, extracted_company_name in iterate_review_pages(url):
        all_reviews.extend(reviews)
        company_name = extracted_company_name  # Update with the actual company name

    save_reviews_to_csv(all_reviews, company_name)