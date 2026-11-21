# BG-REL — review thực thi và candidate R4

Ngày báo cáo: 07/11/2026. Đây là báo cáo của candidate mới; không thay thế các kết quả thực nghiệm đã đóng băng.

## Kết luận

Snapshot đính kèm đúng bản R3: `code.zip` có SHA-256 `eb54f66c36ffa78092b9784b088c2fba11e0b9dfb1761cbb026f99223da6af71`, 32 Python sources, 12.052 dòng (snapshot R3; candidate V2 có 33 sources). Hash từng source khớp Appendix C của handoff. ZIP chứa 3 cache `.pyc`; package candidate không phân phối cache.

Paper PDF 6 trang có cả validation và held-out internal TEST. Các kết quả TEST trong Tables V–VI không mâu thuẫn với một báo cáo validation cũ được lập trước TEST. Kết quả mạnh nhất là tăng REM trong subset VAL match được; paper không chứng minh A3 cải thiện discrimination trên TEST.

Đã có code candidate và kiểm thử thực tế. **Chưa có bằng chứng đủ để gọi toàn bộ R4 hay full paper reproduction là hoàn tất.** Cụ thể, không có DICOM, frozen CSV, checkpoint, full run metadata và archived predictions trong 4 attachments để chạy các kiểm tra thực địa. Hướng dẫn một lượt Windows giải quyết nhu cầu này, không yêu cầu upload dữ liệu lớn.

## Nguồn đã đối chiếu

- `code.zip` đính kèm: source hiện tại, không gán nhầm thành source nguyên bản đã training.
- `AI2007_Group_6 (1).pdf` đính kèm: Methods, Table I–VI, Fig. 1–3, Discussion và reproducibility statement. Đã extract text, xem ảnh trang có các bảng kết quả chính.
- Hai Markdown đính kèm: audit trước và master handoff. Chúng là mô tả/claim lịch sử, không thay cho kết quả runtime mới.
- `bgrel_b0_provenance_search.zip`: đã lấy lại source B0 bằng bytes và xác minh đúng SHA canonical.
- `BGREL_P4_Regression_Pass_and_Adapter_Preflight.zip`: đã lấy lại `test_rel_adapter.py`, gồm 13 test; lưu riêng trong `tests/`.

Không download paper tham khảo bên ngoài, không đánh giá lại tính đúng của từng reference. Nhiệm vụ lượt này là source/paper consistency và refactor an toàn.

## Phát hiện bổ sung so với handoff

**SHA PSPNet trong §3.2 của handoff thiếu một ký tự**, chỉ có 63 hexadecimal characters:

```text
Handoff: 019b167eac6b729fc1bb92bbc185fc1730aaa65819f4e3fe718186cadc044fc
Source:  019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc
```

Source có 64 ký tự, được khai báo tại `generate_lung_masks.py:30` của snapshot. Lượt review laptop dùng giá trị source; chưa có checkpoint ở môi trường này để tự chứng nhận bytes trọng số. Không sửa checkpoint hoặc historical manifest.

## Mapping nghiên cứu và source

| Giai đoạn | Producer/consumer hiện hành | Hành vi cần giữ |
|---|---|---|
| P0 | `audit_data` | Calculated labels, mapping SOP/Study/Series, UID/patient checks, boxes XYXY chưa round |
| P1 | `split_patients` | Patient-level grouping, 80/10/10 frozen; không chia lại |
| P2 | `generate_lung_masks`, QA/pilot/policy, `build_train_lung_masks` | PSPNet channels 4/5; technical geometry screen, không gọi là clinical QA |
| P3 | `sample_controls`, `rel_adapter` | Same-image matching, K0/1/2, ordering/budget/thresholds, frozen trace |
| B0 | historical canonical trainer | Run `b0_dev_s42`, train photometric augmentation; active trainer T2-A không thay thế nó |
| B1 | `train_ablations`, `training_objectives` | Image BCE + lesion-box positive BCE, average lesion→eligible image |
| A3 | `train_bgrel`, `losses`, `regions` | Listwise REL, shared head, fractional ROI, skip K0 trong auxiliary |
| P8a | `evaluate_p8a`, `evaluate_b1_val`, `p8a_bootstrap` | VAL classification và paired patient bootstrap |
| P8b | `evaluate_regions`, `evaluate_b1_regions`, `evaluate_lesion` | REM, Top1, Delta-L/C, blur/fill/RIE, patient-equal contrasts |
| P9 | `evaluate_final_test` | Separate final TEST boundary, epoch-3 checkpoints, FP32, no threshold tuning |

`SOURCE_API_MAP.csv` liệt kê toàn bộ định nghĩa top-level, nơi đến và import nội bộ. `FILE_MAPPING.csv` liệt kê từng source, SHA trước/sau và snapshot giữ lại. CLI độc lập vẫn được giữ dù không có caller import.

## Candidate thay đổi gì, vì sao?

### Cấu hình

Chuyển `AuditPaths` và `get_audit_paths` vào `paths.py`; sửa 10 import statements liên quan đến settings theo source thật, không duy trì hai module config trùng trách nhiệm. `audit_data` ghi nhận hash `paths.py`, không tham chiếu file settings đã retire. Resolver DICOM giữ nguyên body; chỉ ánh xạ đường dẫn vật lý khi đọc, không sửa giá trị frozen CSV. Tên hai JSON dài giữ nguyên.

### Training

Tách nguyên 7 function AST khỏi trainer: `collect_region_scores`, `compute_training_loss`, `candidate_axis`, `geometry_candidates`, `sample_geometry_controls`, `geometry_csv_text`, `text_sha256`. B1 vẫn không dùng controls; A2 geometry-only vẫn là extension chưa được report chạy trong paper. Trainer vẫn tái export tên cũ qua import.

`training_objectives.py` có 440 dòng. Trainer giảm 2.173→1.509 dòng, gồm tách function và Black formatting. Không nói rằng toàn bộ giảm dòng là code duplication được loại bỏ. Module vẫn còn lớn: extraction geometry index, shared training engine và QA orchestration rộng hơn **chưa hoàn thành trong candidate này**.

Hai trainer dùng chung `write_metrics_atomic`, nhưng truyền lại original atomic replacement callback. Vì thế cột, LF bytes, `None` serialization và `.tmp` recovery giữ nguyên. Không hợp nhất bừa RNG functions: B1 và A3 có CPU/CUDA handling khác nhau.

Các provenance source lists của trainer mới có thêm helpers, paths và objective module tương ứng. Điều này ghi nhận source mới nếu có một experiment mới sau này; không sửa config/summary/hash lịch sử.

### P0 và B0 provenance

Thêm guard ở cả direct `audit()` và CLI, chặn reconstruction khi có frozen split seal. CLI chặn trước logger và exception report để tránh ghi đè audit report cũ chỉ vì guard phát lỗi. P0 reconstruction vẫn có thể chạy trong workspace riêng không có seal.

Candidate V2 tách B0 và T2-A thành hai runner. Canonical B0 cung cấp recipe của `train_baseline.py`; T2-A có entrypoint `train_t2a_noaug.py` và self-hash filename tương ứng. Các runner candidate thêm guard chặn train/resume của hai archived run IDs, kể cả gọi hàm trực tiếp. Vì thế chúng có SHA mới, không gán lại thành source lịch sử. Hai bản gốc nguyên byte được giữ riêng, có `BASELINE_SOURCE_VERSIONS.json` đối chiếu.

P6 bỏ việc khóa nhầm bản T2-A dưới filename B0 và thay bằng `verify_current_baseline_layout()`, pin riêng hai SHA hiện tại và kiểm RUN_ID/self-hash/augmentation. **Không bỏ `check_b0_recipe` hoặc thay dataset SHA trong historical config.** P6 vẫn fail nếu dataset historical và active R3 chưa được đối chiếu đúng phiên bản; đây là giới hạn còn lại được report rõ, không phải P6 PASS.

`check_b0_snapshot.py` bổ sung vào lượt Windows kiểm source B0 gốc, recorded trainer SHA, recipe và checkpoint. Tên kết luận là identity/snapshot check; không chứng nhận full replay.

### Chức năng không retire

Giữ `p8a_bootstrap.py`, `evaluate_lesion.py`, hai pilot files, full TRAIN mask builder và TEST evaluator. Cả 23 CLI vẫn hỗ trợ `--help`. Không nhập TEST evaluator vào training hoặc VAL orchestration.

## Bằng chứng kiểm thử tại môi trường này

Môi trường thực chạy: Linux, Python 3.12, torch 2.9.1+cpu, torchvision 0.24.1+cpu. Đây không phải laptop RTX 4050 và không chứng minh BF16/CUDA parity.

| Kiểm tra | Kết quả quan sát | Phạm vi |
|---|---|---|
| Source hash snapshot | PASS | ZIP và từng `.py` so với handoff |
| 42 unittest/parity tests | PASS | Log `verification/final_regression.log` |
| Parse all Python package | PASS | 75 files gồm active, historical, tests, scripts |
| Import 33 candidate modules | PASS | Không chạy phase producers |
| 23 CLI `--help` | PASS | Từ cwd ngoài package; không gọi preflight/run/train |
| 7 extracted functions AST | PASS | Algorithm AST không đổi |
| B1/A2 loss/value/gradient/AdamW | Exact parity | Synthetic fixtures, lambda 0/.02/.06/.1, K0/K1/K2 |
| Metrics CSV serialization | Exact byte parity | A3 và B1/A2; `.tmp` interrupted artifact preserved |
| ROI/model/REL | PASS | Fractional analytic fixture, OS16/shared head, stable REL và K0 |
| Sampler/mask/intervention | PASS | Synthetic masks, geometry constraints, blur/fill pixels và RIE |
| P0 frozen safeguards | PASS | Direct và CLI; CLI không tạo logger/report |
| Source transaction simulation | PASS | Apply, exact rollback, injected failure after backup |
| Missing-project negative path | Expected FAIL | Tạo báo cáo có FAIL, không nhận là experiment result |
| Windows PowerShell execution | PENDING | Không có `pwsh` trong môi trường này |
| Real DICOM/checkpoint replay | PENDING | Chạy một lượt tại laptop |
| Recompute paper tables | PENDING | Cần archived outputs thật trên laptop |
| Historical evaluator runtime | PENDING | Chưa có đầy đủ source/environment pin |

13 restored tests kiểm adapter; 29 tests bổ sung kiểm core, refactor và baseline versions. Không tạo metric nghiên cứu giả. Log snapshot là log tại thời điểm test, không phải chứng nhận về dữ liệu người dùng.

## Source immutability và historical replay

`reproducibility/r3_source/` là snapshot R3 mà người dùng vừa gửi. **Nó không tự trở thành paper-source-immutable.** Source training B0 canonical được khôi phục riêng, có manifest chỉ rõ giới hạn. Không có đầy đủ canonical B1/A3/P8/P9 snapshots ở đây.

Active R3 decoder và sampler đã khác source recorded trong một số run manifests. Candidate giữ strict historical verifiers; một số legacy preflight có thể tiếp tục báo source drift. Đây là block có chủ đích cần giải quyết bằng source đúng phiên bản hoặc modern equivalence track có provenance mới, không phải lý do thay expected SHA.

Laptop reviewer tìm source theo recorded SHA trong package và thư mục backup pre-R3 đã biết. Kết quả `historical_sources` chỉ là source resolution; runtime replay vẫn `PENDING` cho tới khi nguồn được xem và chạy đúng môi trường. Installer yêu cầu `historical_runtime_replay=VERIFIED` cùng candidate hash đã nghiệm thu, nên package này không tự nhận đủ điều kiện cutover.

## Quy trình máy người dùng và cleanup

`RUN_REVIEW.ps1` khởi chạy một bộ kiểm tra, lưu output ngoài project rồi ZIP các bằng chứng nhỏ. Không tải DICOM/checkpoint/prediction rows về assistant. Script được thiết kế không ghi project, dùng `-B`/`PYTHONDONTWRITEBYTECODE` và chỉ đọc 4 DICOM đã chọn từ TRAIN/VAL. Các phép bootstrap đọc archived CSV, không thực hiện TEST inference.

`results_inventory.csv` chỉ có metadata, hashes cho các file text nhỏ, classification và `NO_CHANGE`. Không đánh dấu `results/reports` là rác chỉ dựa trên tên. Bản RSNA root hoặc `.r3b_rsna_stage` chỉ kiểm presence; không hash lại toàn cây hay di chuyển dữ liệu. Archive decisions cuối cùng cần đối chiếu inventory và dependencies.

Transaction code chỉ chạm `code/`. Original tree được rename vào backup ngoài project; candidate staging có checksum; lỗi sau backup tự restore. Rollback từ chối nếu active source đã được chỉnh sau cutover. Sau một cutover đã được nghiệm thu, post-deploy unit suite thất bại sẽ tự rollback. Đây mới là simulation Linux; chưa tuyên bố Windows deployment PASS.

## Các mục còn phải làm để đóng nhiệm vụ

1. Nhận ZIP từ lượt Windows, kiểm metadata/traceback/source resolution và numeric comparisons thật.
2. Khôi phục/match source lịch sử thiếu, đặc biệt evaluator regions và dataset gốc; pin environment thay vì sửa historical records.
3. Hoàn tất version-aware historical replay hoặc một track equivalence minh bạch; không gọi modern source là original paper source.
4. Chốt có cần merge thêm VAL/QA orchestration trong cùng candidate; chỉ retire khi có producer→replacement mapping đủ test.
5. Nghiệm thu candidate, deploy source có backup, post-deploy test và rollback thực tế.
6. Theo inventory thật, chọn archive recoverable các file phụ; giữ paper outputs và provenance. Không xóa vĩnh viễn trong lượt này.

Không yêu cầu chạy lại training, thay split, đổi lambda/threshold hay mở TEST pixels để làm các việc trên.


## Cập nhật V2 theo hai attachments bổ sung

Đã xác minh toàn bộ 16 entries trong manifest của `BGREL_B0_CODE_AND_MASTER_HANDOFF_2026-11-07.zip`. Handoff mới giữ nội dung cũ và thêm phụ lục B0/T2-A; không cung cấp kết quả Windows, predictions hay checkpoint mới. Thông tin này đủ để sửa candidate naming/identity và run protection, nhưng không đủ để tuyên bố P6/full historical replay đã pass.

Bản V2 thay thế gói candidate trước cho lần review tiếp theo; không phải một patch cần cài chồng. Snapshot R3 để đối chiếu live code vẫn giữ nguyên. Không có thay đổi về loss, lambda, split, sampler, metric hay kết quả đã công bố.

---

## 2026-11-07 R4 clean overlay (supersedes V2 candidate status where different)

This file was originally authored for R4 V2. The later consolidated candidate also merges the duplicated BCE train implementation. Active `code/` remains 33 modules but its total line count is 11,962. T2-A is a thin 55-line wrapper over the shared B0 engine; semantic differences in augmentation, run ID, config, validation and safe archive protection are explicitly preserved. `refactor_helpers.verify_current_baseline_layout()` pins the new candidate hashes and must not be substituted for historical dataset/evaluator source SHA validation. Exact sources for the 13 historical recorded hashes are now bundled under `reproducibility/historical_verified/`. See `docs/TEST_REPORT.md`, `docs/FILE_MAPPING.csv`, `docs/CHANGELOG.md`, and `docs/DEPLOYMENT_SCOPE.md` for the final state. The Windows R4 clean install has NOT yet been run; historical runtime replay remains NOT_RUN.
