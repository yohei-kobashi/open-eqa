import json
import re
import os

def main():
    # open-eqaのデータを読み込み
    openEQA = json.load(open("open-eqa-v0.json", encoding="utf-8"))
    
    for data in openEQA:
        m = re.match(r".*/\d+-(\w+)-(.*)", data["episode_history"])
        if m:
            dataset = m.group(1)
            scene_id = m.group(2)
            # scene typeを取得
            if dataset == "scannet":
                txt_path = f"raw/scannet/scans/{scene_id}/{scene_id}.txt"
                if os.path.exists(txt_path):
                    config = open(txt_path, encoding="utf-8").read()
                    m2 = re.search(r"sceneType = (.*)", config)
                    if m2:
                        data["sceneType"] = m2.group(1).strip()
                    else:
                        print("no scenetype: ", scene_id)
                else:
                    print("no config file:", scene_id)
    
    # sceneTypeを追加したjsonを保存
    json.dump(openEQA, open("open-eqa-v0_sceneType.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
