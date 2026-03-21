from scripts.transform import (
    thai_to_arabic,
    only_thai_numbers, 
    html_to_dataframe, 
    delete_columns_from_dataframe,
    apply_to_column
)

__REAL_TEXT_FROM_OCR_A_FILE_PARTY_LIST = """
ส.ส. ๖/๑ (บช)
<page_number>- ๒ -</page_number><table><tr><td>หมายเลขของบัญชีรายชื่อของ พรรคการเมือง</td><th colspan="2">สังกัด<td>ได้คะแนน<br/>(ให้กรอกทั้งตัวเลขและตัวอักษร)</td></th></tr><tr><td></td><td colspan="2">พรรคการเมือง</td><td></td></tr><tr><td>๑</td><td colspan="2">ไทยทรัพย์ทวี</td><td>๖๖ (หกสิบหก)</td></tr><tr><td>๒</td><td colspan="2">เพื่อชาติไทย</td><td>๑๓๐ (หนึ่งร้อยสามสิบ)</td></tr><tr><td>๓</td><td colspan="2">ใหม่</td><td>๓๗ (สามสิบเจ็ด)</td></tr><tr><td>๔</td><td colspan="2">มิติใหม่</td><td>๓๕ (สามสิบห้า)</td></tr><tr><td>๕</td><td colspan="2">รวมใจไทย</td><td>๒๒๖ (สองร้อยยี่สิบหก)</td></tr><tr><td>๖</td><td colspan="2">รวมไทยสร้างชาติ</td><td>๑,๘๕๗ (หนึ่งพันแปดร้อยห้าสิบเจ็ด)</td></tr><tr><td>๗</td><td colspan="2">พลวัต</td><td>๑๐๓ (หนึ่งร้อยสาม)</td></tr><tr><td>๘</td><td colspan="2">ประชาธิปไตยใหม่</td><td>๓๒๓ (สามร้อยยี่สิบสาม)</td></tr><tr><td>๙</td><td colspan="2">เพื่อไทย</td><td>๖,๓๙๘ (หกพันสามร้อยเก้าสิบแปด)</td></tr><tr><td>๑๐</td><td colspan="2">ทางเลือกใหม่</td><td>๓๐๐ (สามร้อย)</td></tr><tr><td>๑๑</td><td colspan="2">เศรษฐกิจ</td><td>๑,๒๙๒ (หนึ่งพันสองร้อยเก้าสิบสอง)</td></tr><tr><td>๑๒</td><td colspan="2">เสรีรวมไทย</td><td>๓๗๔ (สามร้อยเจ็ดสิบสี่)</td></tr><tr><td>๑๓</td><td colspan="2">รวมพลังประชาชน</td><td>๑๖๐ (หนึ่งร้อยหกสิบ)</td></tr><tr><td>๑๔</td><td colspan="2">ท้องที่ไทย</td><td>๗ (เจ็ด)</td></tr><tr><td>๑๕</td><td colspan="2">อนาคตไทย</td><td>๒๙ (ยี่สิบเก้า)</td></tr><tr><td>๑๖</td><td colspan="2">พลังเพื่อไทย</td><td>๔๗ (สี่สิบเจ็ด)</td></tr><tr><td>๑๗</td><td colspan="2">ไทยชนะ</td><td>๔๗ (สี่สิบเจ็ด)</td></tr><tr><td>๑๘</td><td colspan="2">พลังคมใหม่</td><td>๑๐ (สิบ)</td></tr><tr><td>๑๙</td><td colspan="2">สังคมประชาธิปไตยไทย</td><td>๘ (แปด)</td></tr><tr><td>๒๐</td><td colspan="2">ฟิวซัน</td><td>๑๗ (สิบเจ็ด)</td></tr><tr><td>๒๑</td><td colspan="2">ไทรวมพลัง</td><td>๒๒ (ยี่สิบสอง)</td></tr><tr><td>๒๒</td><td colspan="2">ก้าวล่วง</td><td>๘ (แปด)</td></tr><tr><td>๒๓</td><td colspan="2">ปวงชนไทย</td><td>๔๓ (สี่สิบสาม)</td></tr><tr><td>๒๔</td><td colspan="2">วิชชินใหม่</td><td>๑๔ (สิบสี่)</td></tr><tr><td>๒๕</td><td colspan="2">เพื่อชีวิตใหม่</td><td>๗ (เจ็ด)</td></tr><tr><td>๒๖</td><td colspan="2">คลองไทย</td><td>๓๙ (สามสิบเก้า)</td></tr><tr><td>๒๗</td><td colspan="2">ประชาธิปัตย์</td><td>๑๑,๗๙๕ (หนึ่งหมื่นหนึ่งพันเจ็ดร้อยเก้า สิบห้า)</td></tr><tr><td>๒๘</td><td colspan="2">ไทยก้าวหน้า</td><td>๔๕ (สี่สิบห้า)</td></tr><tr><td>๒๙</td><td colspan="2">ไทยภักดี</td><td>๒,๑๗๓ (สองพันหนึ่งร้อยเจ็ดสิบสาม)</td></tr><tr><td>๓๐</td><td colspan="2">แรงงานสร้างชาติ</td><td>๑๔ (สิบสี่)</td></tr><tr><td>๓๑</td><td colspan="2">ประชากรไทย</td><td>๔๐ (สี่สิบ)</td></tr></table>
"""

__REAL_TEXT_FROM_OCR_A_FILE_CONSITUENCY_LIST = """
ส.ส. ๖/๑

<table><tr><td>หมายเลข<br/>ประจำตัว<br/>ผู้สมัคร</td><td>ชื่อตัว - ชื่อสกุล<br/>ผู้สมัครรับเลือกตั้ง</td><td>สังกัด<br/>พรรคการเมือง</td><th colspan="2">ได้คะแนน<br/>(ให้กรอกทั้งตัวเลขและตัวอักษร)</th></tr><tr><td>๕</td><td>นายปารเมศ วิทยารักษ์สรรค์</td><td>ประชาชน</td><td colspan="2">๓๔,๑๖๗ (สามหมื่นสี่พันหนึ่งร้อยหกสิบเจ็ด)</td></tr><tr><td>๙</td><td>นายพีรวุฒิ พิมพ์สมฤดี</td><td>ประชาธิปัตย์</td><td colspan="2">๑๔,๘๑๓ (หนึ่งหมื่นสี่พันแปดร้อยสิบสาม)</td></tr><tr><td>๑</td><td>นางสาวลลิดา เพรศิวัฒนา</td><td>ภูมิใจไทย</td><td colspan="2">๑๔,๓๖๘ (หนึ่งหมื่นสี่พันสามร้อยหกสิบแปด)</td></tr><tr><td>๘</td><td>นายญาณกิตติ์ หว่งทรัพย์</td><td>เพื่อไทย</td><td colspan="2">๖,๐๓๐ (หกพันสามสิบ)</td></tr><tr><td>๖</td><td>นายพลัฏฐ์ ศิริกุลพิสุทธิ์</td><td>รวมไทยสร้างชาติ</td><td colspan="2">๒,๐๗๕ (สองพันเจ็ดสิบห้า)</td></tr><tr><td>๑๒</td><td>นายแทนคุณ จิตอิสระ</td><td>โอกาสใหม่</td><td colspan="2">๑,๑๓๓ (หนึ่งพันหนึ่งร้อยสามสิบสาม)</td></tr><tr><td>๗</td><td>นายภัทรพล หมดมลทิน</td><td>ไทยภักดี</td><td colspan="2">๑,๐๒๓ (หนึ่งพันยี่สิบสาม)</td></tr><tr><td>๒</td><td>พันตำรวจโทหญิง แสนสุข อุทยานินทร์</td><td>เศรษฐกิจ</td><td colspan="2">๙๗๙ (เก้าร้อยเจ็ดสิบเก้า)</td></tr><tr><td>๑๑</td><td>นายพิเชษฐ์ ไทยนิยม</td><td>ไทยสร้างไทย</td><td colspan="2">๖๒๙ (หนึ่งร้อยแปดสิบเก้า)</td></tr><tr><td>๑๖</td><td>นายกานต์ กิตติอำพน</td><td>ไทยก้าวใหม่</td><td colspan="2">๔๘๙ (สี่ร้อยแปดสิบเอ็ด)</td></tr><tr><td>๔</td><td>นายธนะสิทธิ์ ล้อสุวรรณทีปต์</td><td>พลวัต</td><td colspan="2">๓๕๑ (สามร้อยห้าสิบสี่)</td></tr><tr><td>๓</td><td>นายอัครพล คฤหเดชรัตนา</td><td>กล้าธรรม</td><td colspan="2">๒๔๔ (สองร้อยสี่สิบสี่)</td></tr><tr><td>๑๐</td><td>นายวีระยุทธ ประกอบการ</td><td>ปวงชนไทย</td><td colspan="2">๑๖๘ (หนึ่งร้อยหกสิบแปด)</td></tr><tr><td>๑๕</td><td>นายอดัม ชินรัตนพิสิทธิ์</td><td>รักชาติ</td><td colspan="2">๑๖๕ (หนึ่งร้อยห้าสิบสี่)</td></tr><tr><td>๑๗</td><td>นายประสงค์ ประสพโชค</td><td>ทางเลือกใหม่</td><td colspan="2">๑๑๓ (หนึ่งร้อยสิบสาม)</td></tr><tr><td>๑๓</td><td>นายเลิศชาย สุจริตกุล</td><td>วิชชั่นใหม่</td><td colspan="2">๑๕๔ (เก้าสิบสี่)</td></tr><tr><td>๑๔</td><td>นายมงคล เสมอภาพ</td><td>ประชาธิปไตยใหม่</td><td colspan="2">๙๔ (แปดสิบ)</td></tr><tr><td>๑๘</td><td>นายกิตติคุณ ซื่อนแย้ม</td><td>พลังประชารัฐ</td><td colspan="2">๘๐ (แปดสิบ)</td></tr><tr><td colspan="3">รวมคะแนนทั้งสิ้น</td><td colspan="2">๗๗,๐๗๕ (เจ็ดหมื่นเจ็ดพันเจ็ดสิบห้า)</td></tr></table>

PageNumber: - ๒ -
"""

def main():
    df = html_to_dataframe(__REAL_TEXT_FROM_OCR_A_FILE_PARTY_LIST)
    df = delete_columns_from_dataframe(df, 0, 1)

    # as we can see, the voted numbers are in thai with thai text in parentheses,
    # so we need to filter out the thai numbers and convert them to arabic numbers
    # actually, pandas has a built-in function to apply or filter but i just want to show off my skills in writing a custom function to do that :)
    # we will act the pandas is just a dataframe and we will apply the function to the last column, which is the votes column
    df = apply_to_column(df=df, col_index=-1, func=only_thai_numbers)
    df = apply_to_column(df=df, col_index=-1, func=thai_to_arabic)
    print(df)


if __name__ == "__main__":
    main()