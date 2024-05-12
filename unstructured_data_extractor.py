import re
import os
from typing import List

from base_component import BaseComponent
from llm.basellm import BaseLLM
from unstructured_data_utils import (
    nodesTextToListOfDict,
    relationshipTextToListOfDict,
)


def generate_system_message_with_schema() -> str:
    return """
Bạn là một nhà khoa học dữ liệu đang làm việc trong lĩnh vực pháp luật. Nhiệm vụ của bạn là trích xuất thông tin từ các văn bản luật và chuyển đổi chúng thành một cơ sở dữ liệu đồ thị.
Hãy tập trung vào việc xác định và trích xuất các đối tượng quan trọng sau đây từ văn bản luật:
Đất đai (ĐấtĐai): Bao gồm các loại đất khác nhau.
Chủ sở hữu đất đai (ChủSởHữuĐấtĐai): Bao gồm Nhà nước đại diện chủ sở hữu toàn dân và các chủ sở hữu khác.
Người sử dụng đất (NgườiSửDụngĐất): Bao gồm công dân, tổ chức trong nước, người Việt Nam định cư ở nước ngoài, tổ chức nước ngoài, cá nhân nước ngoài.
Cơ quan nhà nước (CơQuanNhàNước): Gồm các cơ quan thực hiện quyền hạn và trách nhiệm đại diện chủ sở hữu toàn dân về đất đai, thực hiện nhiệm vụ thống nhất quản lý nhà nước về đất đai.
Quyền và nghĩa vụ (QuyềnVàNghĩaVụ): Liên quan đến quyền và nghĩa vụ của các chủ thể có liên quan đến đất đai.
Văn bản pháp luật (VănBảnPhápLuật): Các loại giấy tờ pháp luật.
Các hoạt động liên quan đến đất đai (HoạtĐộngLiênQuanĐếnĐấtĐai): Như bảo vệ, cải tạo, phục hồi đất, quản lý và sử dụng đất.
Hãy cung cấp một tập hợp các Node dưới dạng [ENTITY, TYPE, PROPERTIES] và một tập hợp các relationship dưới dạng [ENTITY1, RELATIONSHIP, ENTITY2, PROPERTIES].
Hãy chú ý đến kiểu dữ liệu của các thuộc tính (property), nếu bạn không thể tìm thấy dữ liệu cho một thuộc tính, hãy đặt giá trị của nó là null. Không tự tạo ra bất kỳ dữ liệu nào và không thêm bất kỳ dữ liệu bổ sung nào. Nếu bạn không thể tìm thấy bất kỳ dữ liệu nào cho một node hoặc relationship, đừng thêm nó vào.
Chỉ thêm các node và relationship có trong schema. Nếu bạn không nhận được bất kỳ relationship nào trong schema, chỉ thêm các node.
Ví dụ:
Schema:
Nodes: [VănBảnPhápLuật {số_hiệu: string, ngày_ban_hành: date, ngày_có_hiệu_lực: date}, CơQuanNhàNước {tên: string}]
Relationships: [VănBảnPhápLuật, đượcBanHànhBởi, CơQuanNhàNước]
Dữ liệu: Luật số 50/2019/QH14 về Chứng khoán được Quốc hội ban hành ngày 26 tháng 11 năm 2019 và có hiệu lực từ ngày 01 tháng 01 năm 2021.
Nodes:
[
["luật_50_2019_QH14", "VănBảnPhápLuật", {"số_hiệu": "50/2019/QH14", "ngày_ban_hành": "2019-11-26", "ngày_có_hiệu_lực": "2021-01-01"}],
["quốc_hội", "CơQuanNhàNước", {"tên": "Quốc hội"}]
]
Relationships:
[
["luật_50_2019_QH14", "đượcBanHànhBởi", "quốc_hội", {}]
]
"""


def generate_system_message() -> str:
    return """
Bạn là một nhà khoa học dữ liệu đang làm việc trong lĩnh vực pháp luật. Nhiệm vụ của bạn là trích xuất thông tin từ các văn bản luật và chuyển đổi chúng thành một cơ sở dữ liệu đồ thị.
Hãy tập trung vào việc xác định và trích xuất các đối tượng quan trọng sau đây từ văn bản luật:
Đất đai (ĐấtĐai): Bao gồm các loại đất khác nhau.
Chủ sở hữu đất đai (ChủSởHữuĐấtĐai): Bao gồm Nhà nước đại diện chủ sở hữu toàn dân và các chủ sở hữu khác.
Người sử dụng đất (NgườiSửDụngĐất): Bao gồm công dân, tổ chức trong nước, người Việt Nam định cư ở nước ngoài, tổ chức nước ngoài, cá nhân nước ngoài.
Cơ quan nhà nước (CơQuanNhàNước): Gồm các cơ quan thực hiện quyền hạn và trách nhiệm đại diện chủ sở hữu toàn dân về đất đai, thực hiện nhiệm vụ thống nhất quản lý nhà nước về đất đai.
Quyền và nghĩa vụ (QuyềnVàNghĩaVụ): Liên quan đến quyền và nghĩa vụ của các chủ thể có liên quan đến đất đai.
Văn bản pháp luật (VănBảnPhápLuật): Các loại giấy tờ pháp luật.
Các hoạt động liên quan đến đất đai (HoạtĐộngLiênQuanĐếnĐấtĐai): Như bảo vệ, cải tạo, phục hồi đất, quản lý và sử dụng đất.
Hãy cung cấp một tập hợp các Node dưới dạng [ENTITY_ID, TYPE, PROPERTIES] và một tập hợp các relationship dưới dạng [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
Điều quan trọng là ENTITY_ID_1 và ENTITY_ID_2 phải tồn tại dưới dạng các node với ENTITY_ID khớp. Nếu bạn không thể ghép một relationship với một cặp node, đừng thêm nó vào.
Khi bạn tìm thấy một node hoặc relationship mà bạn muốn thêm, hãy cố gắng tạo một TYPE chung cho nó để mô tả entity. Bạn cũng có thể coi nó như một nhãn.
Ví dụ:
Dữ liệu: Luật số 45/2013/QH13 của Quốc hội về Đất đai được ban hành ngày 29 tháng 11 năm 2013 và có hiệu lực từ ngày 01 tháng 07 năm 2014. Luật này quy định về quyền hạn, trách nhiệm của Nhà nước đại diện chủ sở hữu toàn dân về đất đai và thống nhất quản lý về đất đai; chế độ quản lý và sử dụng đất đai; quyền và nghĩa vụ của người sử dụng đất.
Nodes:
["luật_45_2013", "VănBảnPhápLuật", {"tên_văn_bản": "Luật Đất đai", "số_hiệu": "45/2013/QH13", "ngày_ban_hành": "2013-11-29", "ngày_có_hiệu_lực": "2014-07-01"}],
["quốc_hội", "CơQuanNhàNước", {"tên_cơ_quan": "Quốc hội"}],
["đất_đai", "ĐấtĐai", {}],
["nhà_nước", "ChủSởHữuĐấtĐai", {"tên_chủ_sở_hữu": "Nhà nước", "đại_diện_chủ_sở_hữu": "toàn dân"}],
["người_sử_dụng_đất", "NgườiSửDụngĐất", {}],
["quyền_của_người_sử_dụng_đất", "QuyềnVàNghĩaVụ", {"loại_quyền_nghĩa_vụ": "Quyền của người sử dụng đất"}],
["nghĩa_vụ_của_người_sử_dụng_đất", "QuyềnVàNghĩaVụ", {"loại_quyền_nghĩa_vụ": "Nghĩa vụ của người sử dụng đất"}]
Relationships:
["luật_45_2013", "đượcBanHànhBởi", "quốc_hội", {"ngày_ban_hành": "2013-11-29"}],
["luật_45_2013", "quyĐịnhVề", "đất_đai", {}],
["nhà_nước", "làChủSởHữu", "đất_đai", {"đại_diện_chủ_sở_hữu": "toàn dân"}],
["luật_45_2013", "quyĐịnhVề", "quyền_của_người_sử_dụng_đất", {}],
["luật_45_2013", "quyĐịnhVề", "nghĩa_vụ_của_người_sử_dụng_đất", {}],
["người_sử_dụng_đất", "cóQuyền", "quyền_của_người_sử_dụng_đất", {}],
["người_sử_dụng_đất", "cóNghĩaVụ", "nghĩa_vụ_của_người_sử_dụng_đất", {}]
"""


def generate_system_message_with_labels() -> str:
    return """
Bạn là một nhà khoa học dữ liệu đang làm việc trong lĩnh vực pháp luật. Nhiệm vụ của bạn là trích xuất thông tin từ các văn bản luật và chuyển đổi chúng thành một cơ sở dữ liệu đồ thị.
Hãy tập trung vào việc xác định và trích xuất các đối tượng quan trọng sau đây từ văn bản luật:
Đất đai (ĐấtĐai): Bao gồm các loại đất khác nhau.
Chủ sở hữu đất đai (ChủSởHữuĐấtĐai): Bao gồm Nhà nước đại diện chủ sở hữu toàn dân và các chủ sở hữu khác.
Người sử dụng đất (NgườiSửDụngĐất): Bao gồm công dân, tổ chức trong nước, người Việt Nam định cư ở nước ngoài, tổ chức nước ngoài, cá nhân nước ngoài.
Cơ quan nhà nước (CơQuanNhàNước): Gồm các cơ quan thực hiện quyền hạn và trách nhiệm đại diện chủ sở hữu toàn dân về đất đai, thực hiện nhiệm vụ thống nhất quản lý nhà nước về đất đai.
Quyền và nghĩa vụ (QuyềnVàNghĩaVụ): Liên quan đến quyền và nghĩa vụ của các chủ thể có liên quan đến đất đai.
Văn bản pháp luật (VănBảnPhápLuật): Các loại giấy tờ pháp luật.
Các hoạt động liên quan đến đất đai (HoạtĐộngLiênQuanĐếnĐấtĐai): Như bảo vệ, cải tạo, phục hồi đất, quản lý và sử dụng đất.
Hãy cung cấp một tập hợp các Node dưới dạng [ENTITY, TYPE, PROPERTIES] và một tập hợp các relationship dưới dạng [ENTITY1, RELATIONSHIP, ENTITY2, PROPERTIES].
Hãy chú ý đến kiểu dữ liệu của các thuộc tính (property), nếu bạn không thể tìm thấy dữ liệu cho một thuộc tính, hãy đặt giá trị của nó là null. Không tự tạo ra bất kỳ dữ liệu nào và không thêm bất kỳ dữ liệu bổ sung nào. Nếu bạn không thể tìm thấy bất kỳ dữ liệu nào cho một node hoặc relationship, đừng thêm nó vào.
Điều quan trọng là ENTITY_ID_1 và ENTITY_ID_2 phải tồn tại dưới dạng các node với ENTITY_ID khớp. Nếu bạn không thể ghép một relationship với một cặp node, đừng thêm nó vào.
Khi bạn tìm thấy một node hoặc relationship mà bạn muốn thêm, hãy cố gắng tạo một TYPE chung cho nó để mô tả entity. Bạn cũng có thể coi nó như một nhãn.
Bạn sẽ được cung cấp một danh sách các type mà bạn nên cố gắng sử dụng khi tạo TYPE cho một node. Nếu bạn không thể tìm thấy một type phù hợp với node, bạn có thể tạo một type mới.
Ví dụ:
Types: ["VănBảnPhápLuật", "CơQuanNhàNước", "ĐấtĐai", "ChủSởHữuĐấtĐai", "NgườiSửDụngĐất", "QuyềnVàNghĩaVụ"]
Dữ liệu: Luật số 50/2019/QH14 về Chứng khoán được Quốc hội ban hành ngày 26 tháng 11 năm 2019 và có hiệu lực từ ngày 01 tháng 01 năm 2021.
Nodes:
[
["luật_50_2019_QH14", "VănBảnPhápLuật", {"số_hiệu": "50/2019/QH14", "ngày_ban_hành": "2019-11-26", "ngày_có_hiệu_lực": "2021-01-01"}],
["quốc_hội", "CơQuanNhàNước", {"tên": "Quốc hội"}]
]
Relationships:
[
["luật_50_2019_QH14", "đượcBanHànhBởi", "quốc_hội", {}]
]
"""


def generate_prompt(data) -> str:
    return f"""
Data: {data}"""


def generate_prompt_with_schema(data, schema) -> str:
    return f"""
Schema: {schema}
Data: {data}"""


def generate_prompt_with_labels(data, labels) -> str:
    return f"""
Data: {data}
Types: {labels}"""


def splitString(string, max_length) -> List[str]:
    return [string[i: i + max_length] for i in range(0, len(string), max_length)]


def splitStringToFitTokenSpace(
    llm: BaseLLM, string: str, token_use_per_string: int
) -> List[str]:
    allowed_tokens = llm.max_allowed_token_length() - token_use_per_string
    chunked_data = splitString(string, 500)
    combined_chunks = []
    current_chunk = ""
    for chunk in chunked_data:
        if (
            llm.num_tokens_from_string(current_chunk)
            + llm.num_tokens_from_string(chunk)
            < allowed_tokens
        ):
            current_chunk += chunk
        else:
            combined_chunks.append(current_chunk)
            current_chunk = chunk
    combined_chunks.append(current_chunk)

    return combined_chunks


def getNodesAndRelationshipsFromResult(result):
    regex = r"Nodes:\s+(.*?)\s?\s?Relationships:\s?\s?(.*)"
    internalRegex = r"\[(.*?)\]"
    nodes = []
    relationships = []
    for row in result:
        parsing = re.match(regex, row, flags=re.S)
        if parsing == None:
            continue
        rawNodes = str(parsing.group(1))
        rawRelationships = parsing.group(2)
        nodes.extend(re.findall(internalRegex, rawNodes))
        relationships.extend(re.findall(internalRegex, rawRelationships))

    result = dict()
    result["nodes"] = []
    result["relationships"] = []
    result["nodes"].extend(nodesTextToListOfDict(nodes))
    result["relationships"].extend(relationshipTextToListOfDict(relationships))
    return result


class DataExtractor(BaseComponent):
    llm: BaseLLM

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def process(self, chunk):
        messages = [
            {"role": "system", "content": generate_system_message()},
            {"role": "user", "content": generate_prompt(chunk)},
        ]
        print(messages)
        output = self.llm.generate(messages)
        return output

    def process_with_labels(self, chunk, labels):
        messages = [
            {"role": "system", "content": generate_system_message_with_schema()},
            {"role": "user", "content": generate_prompt_with_labels(
                chunk, labels)},
        ]
        print(messages)
        output = self.llm.generate(messages)
        return output

    def run(self, data: str) -> List[str]:
        system_message = generate_system_message()
        prompt_string = generate_prompt("")
        token_usage_per_prompt = self.llm.num_tokens_from_string(
            system_message + prompt_string
        )
        chunked_data = splitStringToFitTokenSpace(
            llm=self.llm, string=data, token_use_per_string=token_usage_per_prompt
        )

        results = []
        labels = set()
        print("Starting chunked processing")
        for chunk in chunked_data:
            proceededChunk = self.process_with_labels(chunk, list(labels))
            print("proceededChunk", proceededChunk)
            chunkResult = getNodesAndRelationshipsFromResult([proceededChunk])
            print("chunkResult", chunkResult)
            newLabels = [node["label"] for node in chunkResult["nodes"]]
            print("newLabels", newLabels)
            results.append(proceededChunk)
            labels.update(newLabels)

        return getNodesAndRelationshipsFromResult(results)
