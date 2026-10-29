# original

팀원이 처음 준 코드를 **수정 없이 그대로** 보관한 폴더 (비교·참고용).

| 파일 | 내용 |
|---|---|
| `yellow_count_node.py` | 노란색이 나타났다 사라질 때마다 count, 3 이 되면 END + `/cmd_vel` 정지 |

- 이 폴더는 패키지에 설치되지 않음 (`ros2 run` 대상 아님). 실제로 쓰는 노드는 `plate_vision/red_count_node.py`
- 원본과 현재 버전 비교:

```bash
git diff --no-index plate_vision/original/yellow_count_node.py plate_vision/plate_vision/red_count_node.py
```
