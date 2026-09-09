import os
import re
import requests

# Insert your CFBD API key here or load it from your environment
API_KEY = "YOUR_CFBD_API_KEY"
url = "https://api.collegefootballdata.com/teams"
headers = {
    "Authorization": f"Bearer {APIKeyHere}",
    "Accept": "application/json"
}

# Create the local logos directory if it doesn't exist
output_dir = "src/static/logos"
os.makedirs(output_dir, exist_ok=True)

print("Fetching team list from CFBD...")
response = requests.get(url, headers=headers)
response.raise_for_status()
teams = response.json()

count = 0
for team in teams:
    school = team.get("school")
    logos = team.get("logos")
    
    if school and logos and logos[0]:
        logo_url = logos[0]
        
        # Generate a clean, standardized filename matching your template logic
        safe_name = school.lower().replace('&', 'and')
        safe_name = re.sub(r'[^a-z0-9]', '_', safe_name)
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        
        filename = os.path.join(output_dir, f"{safe_name}.png")
        
        try:
            img_res = requests.get(logo_url, timeout=10)
            if img_res.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(img_res.content)
                count += 1
                print(f"[{count}] Saved: {school} -> {filename}")
        except Exception as e:
            print(f"Failed to download logo for {school}: {e}")

print(f"Finished! Downloaded {count} team logos to {output_dir}/.")
