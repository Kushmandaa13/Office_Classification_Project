import os
import requests
import time
from urllib.parse import urlencode

def download_from_flickr(query, num_images, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"  Searching Flickr for '{query}'...")
    
    downloaded = 0
    page = 1
    
    while downloaded < num_images and page <= 10:  # Limit to 10 pages
        try:
            # Flickr public feed (no API key needed)
            url = f"https://www.flickr.com/services/feeds/photos_public.gne?format=json&nojsoncallback=1&tags={query.replace(' ', ',')}&per_page=20&page={page}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                if not items:
                    break
                
                for item in items:
                    if downloaded >= num_images:
                        break
                    
                    try:
                        # Get the medium size image URL
                        img_url = item['media']['m'].replace('_m.jpg', '_b.jpg')  # Get larger version
                        
                        # Download image
                        img_response = requests.get(img_url, timeout=15, stream=True)
                        
                        if img_response.status_code == 200:
                            file_path = os.path.join(output_folder, f"flickr_{downloaded+1}.jpg")
                            
                            with open(file_path, 'wb') as f:
                                for chunk in img_response.iter_content(1024):
                                    f.write(chunk)
                            
                            # Verify file size
                            if os.path.getsize(file_path) > 5000:  # At least 5KB
                                downloaded += 1
                                if downloaded % 10 == 0:
                                    print(f"    Progress: {downloaded}/{num_images}")
                            else:
                                os.remove(file_path)
                        
                    except Exception as e:
                        continue
                
                page += 1
                time.sleep(0.5)  # Be polite to Flickr
                
            else:
                break
                
        except Exception as e:
            print(f"    Error on page {page}: {str(e)}")
            break
    
    return downloaded

def download_from_unsplash(query, num_images, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"  Searching Unsplash for '{query}'...")
    
    downloaded = 0
    page = 1
    per_page = 30
    
    while downloaded < num_images and page <= 5:
        try:
            # Unsplash Source API (no authentication needed for basic use)
            # We'll use their direct image URLs
            base_url = "https://source.unsplash.com/1600x900/?"
            
            for i in range(per_page):
                if downloaded >= num_images:
                    break
                
                try:
                    # Add random parameter to get different images
                    img_url = f"{base_url}{query.replace(' ', ',')}&sig={page}_{i}"
                    
                    response = requests.get(img_url, timeout=15, stream=True, allow_redirects=True)
                    
                    if response.status_code == 200:
                        file_path = os.path.join(output_folder, f"unsplash_{downloaded+1}.jpg")
                        
                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)
                        
                        # Verify file size
                        if os.path.getsize(file_path) > 10000:  # At least 10KB
                            downloaded += 1
                            if downloaded % 10 == 0:
                                print(f"    Progress: {downloaded}/{num_images}")
                        else:
                            os.remove(file_path)
                    
                    time.sleep(0.3)  # Rate limiting
                    
                except Exception as e:
                    continue
            
            page += 1
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            break
    
    return downloaded

def download_from_pexels(query, num_images, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"  Searching Pexels for '{query}'...")
    
    # Pexels free API endpoint (works without key for basic access)
    # Using their website's public search
    
    downloaded = 0
    
    # Try to get images from Pexels website directly
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        search_url = f"https://www.pexels.com/search/{query.replace(' ', '%20')}/"
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Simple regex to find image URLs in the HTML
            import re
            pattern = r'https://images\.pexels\.com/photos/\d+/[^"\']*\.jpeg\?[^"\']+'
            matches = re.findall(pattern, response.text)
            
            unique_urls = list(set(matches))[:num_images * 2]  # Get extras
            
            for idx, img_url in enumerate(unique_urls):
                if downloaded >= num_images:
                    break
                
                try:
                    # Download image
                    img_response = requests.get(img_url, headers=headers, timeout=15, stream=True)
                    
                    if img_response.status_code == 200:
                        file_path = os.path.join(output_folder, f"pexels_{downloaded+1}.jpg")
                        
                        with open(file_path, 'wb') as f:
                            for chunk in img_response.iter_content(1024):
                                f.write(chunk)
                        
                        if os.path.getsize(file_path) > 10000:
                            downloaded += 1
                            if downloaded % 10 == 0:
                                print(f"    Progress: {downloaded}/{num_images}")
                        else:
                            os.remove(file_path)
                    
                    time.sleep(0.2)
                    
                except Exception as e:
                    continue
    
    except Exception as e:
        print(f"    Error: {str(e)}")
    
    return downloaded

def main():
    # Classes that need images
    classes_to_download = {
        'Keyboard': 100,
        'Mouse': 100,
        'Mug': 100,
        'Pen_Pencil': 100,
        'Stapler': 100,
        'Tape_Dispenser': 100
    }
    
    base_dir = "data/train"
    
    print("=" * 70)
    print("MULTI-SOURCE IMAGE DOWNLOADER")
    print("=" * 70)
    print("Sources: Flickr + Unsplash + Pexels")
    print("No API keys required!")
    print("=" * 70)
    print()
    
    for idx, (class_name, target_images) in enumerate(classes_to_download.items(), 1):
        print(f"[{idx}/{len(classes_to_download)}] {class_name}")
        
        output_folder = os.path.join(base_dir, class_name)
        search_query = class_name.replace('_', ' ')
        
        total_downloaded = 0
        
        # Try Flickr first (about 1/3 of target)
        flickr_count = download_from_flickr(search_query, target_images // 3, output_folder)
        total_downloaded += flickr_count
        print(f"Flickr: {flickr_count} images")
        
        # Try Pexels (about 1/3 of target)
        if total_downloaded < target_images:
            pexels_count = download_from_pexels(search_query, target_images // 3, output_folder)
            total_downloaded += pexels_count
            print(f"    Pexels: {pexels_count} images")
        
        # Try Unsplash for remaining
        if total_downloaded < target_images:
            remaining = target_images - total_downloaded
            unsplash_count = download_from_unsplash(search_query, remaining, output_folder)
            total_downloaded += unsplash_count
            print(f"    Unsplash: {unsplash_count} images")
        
        print(f"Total downloaded: {total_downloaded}/{target_images} images")
        print()
        
        time.sleep(1)
    
    print("=" * 70)
    print("Download completed!")
    print("Run 'python count.py' to verify your dataset")
    print("=" * 70)

if __name__ == "__main__":
    main()