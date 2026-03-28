import re
from pathlib import Path
from typing import List, Tuple
import math
from collections import defaultdict


class ThaiTokenizer:
    def __init__(self):
        self.thai_pattern = re.compile(r'[\u0E00-\u0E7F]+|[a-zA-Z0-9]+|\d+')
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = text.lower()
        tokens = self.thai_pattern.findall(text)
        return [t for t in tokens if len(t) > 1]


class KnowledgeBase:
    """Knowledge base with TF-IDF based retrieval."""
    
    PRODUCT_ALIASES = {
        # Watch - วงโคจร
        'watch s3 ultra': ['WK-SW-001', 'วงโคจร watch s3 ultra', 'วงโคจร'],
        's3 ultra': ['WK-SW-001', 'วงโคจร watch s3 ultra'],
        'watch s3 pro': ['WK-SW-002', 'วงโคจร watch s3 pro'],
        's3 pro': ['WK-SW-002'],
        'watch s3': ['WK-SW-003', 'วงโคจร watch s3'],
        'watch lite': ['WK-SW-004'],
        'watch se': ['WK-SW-005'],
        'watch kids': ['WK-SW-006'],
        'วงโคจร': ['WK-SW'],
        # Phones - สายฟ้า
        'x9 pro max': ['SF-SP-001', 'สายฟ้า โฟน x9 pro max', 'สายฟ้า'],
        'x9 pro': ['SF-SP-002', 'สายฟ้า โฟน x9 pro'],
        'x9': ['SF-SP-003', 'สายฟ้า โฟน x9'],
        'x9 fe': ['SF-SP-010', 'สายฟ้า โฟน x9 fe'],
        'x8 pro': ['SF-SP-004'],
        'x8': ['SF-SP-005'],
        'x7 pro': ['SF-SP-006'],
        'x7': ['SF-SP-007'],
        'x5': ['SF-SP-008', 'SF-SP-009'],
        'rugged r1': ['SF-SP-015', 'สายฟ้า โฟน rugged r1', 'สายฟ้า rugged'],
        'duopad': ['SF-SP-011', 'สายฟ้า duopad'],
        'สายฟ้า': ['SF-SP'],
        # Laptops - ดาวเหนือ / NovaTech
        'airbook 15 pro': ['DN-LT-001', 'ดาวเหนือ airbook 15 pro'],
        'airbook 15': ['DN-LT-001'],
        'airbook 14 pro': ['DN-LT-002', 'ดาวเหนือ airbook 14 pro'],
        'airbook 14': ['DN-LT-002', 'DN-LT-003', 'ดาวเหนือ airbook 14'],
        'stormbook g7 pro': ['DN-LT-007'],
        'stormbook g7': ['DN-LT-007', 'ดาวเหนือ stormbook g7'],
        'stormbook g5 pro': ['DN-LT-008'],
        'stormbook g5': ['DN-LT-008', 'DN-LT-009', 'ดาวเหนือ stormbook g5'],
        'stormbook': ['DN-LT-007', 'DN-LT-008', 'DN-LT-009'],
        'creatorbook 16 pro': ['DN-LT-014'],
        'creatorbook 16': ['DN-LT-014', 'ดาวเหนือ creatorbook 16'],
        'workstation 17': ['DN-LT-011', 'DN-LT-012'],
        'slimbook 14': ['NT-LT-001', 'novatech slimbook 14'],
        'slimbook 15': ['NT-LT-002', 'novatech slimbook 15'],
        'powerbook 17': ['NT-LT-003'],
        'ดาวเหนือ': ['DN-LT'],
        'novatech': ['NT-LT', 'NT-EB'],
        # Audio - คลื่นเสียง
        'headpro x1': ['KS-HP-001', 'KS-HP-002', 'คลื่นเสียง เฮดโปร x1'],
        'headpro x1 max': ['KS-HP-001'],
        'headpro x1 pro': ['KS-HP-002'],
        'headon 700': ['KS-HP-003', 'คลื่นเสียง เฮดออน 700'],
        'headon 500': ['KS-HP-004', 'คลื่นเสียง เฮดออน 500'],
        'headon 300': ['KS-HP-005', 'KS-HP-006', 'คลื่นเสียง เฮดออน 300'],
        'headon 300 wireless': ['KS-HP-005'],
        'headon 300 wired': ['KS-HP-006'],
        'buds z5 pro': ['KS-EB-001', 'คลื่นเสียง บัดส์ z5 pro'],
        'buds z5': ['KS-EB-002', 'คลื่นเสียง บัดส์ z5'],
        'buds z3 pro': ['KS-EB-003'],
        'sport x': ['KS-EB-004', 'คลื่นเสียง sport x'],
        'sport lite': ['KS-EB-005', 'คลื่นเสียง sport lite'],
        'novabuds pro': ['NT-EB-001', 'novatech novabuds pro'],
        'novabuds lite': ['NT-EB-002'],
        'คลื่นเสียง': ['KS-HP', 'KS-EB', 'KS-SK'],
        # Tablets - สายฟ้า
        'tab s9 pro': ['SF-TB-001', 'สายฟ้า แท็บ s9 pro'],
        'tab s9': ['SF-TB-002', 'สายฟ้า แท็บ s9'],
        'tab a5 pro': ['SF-TB-003'],
        'tab a5': ['SF-TB-003', 'SF-TB-004', 'สายฟ้า แท็บ a5'],
        'tab a5 wifi': ['SF-TB-004'],
        'tab e3 pro': ['SF-TB-005'],
        'tab e3': ['SF-TB-006'],
        'tab draw pro': ['SF-TB-007', 'สายฟ้า แท็บ draw pro'],
        'tab kids': ['SF-TB-008'],
        # Accessories - จุดเชื่อม / พัลส์เกียร์ / ArcWave
        'saifah pen gen 2': ['JC-CS-006', 'ปากกา saifah pen gen 2'],
        'saifah pen': ['JC-CS-006', 'ปากกา saifah pen'],
        'ปากกา saifah': ['JC-CS-006'],
        'qipad 15w': ['JC-CH-005', 'จุดเชื่อม qipad 15w'],
        'qipad 15': ['JC-CH-005'],
        'qipad': ['JC-CH-005', 'JC-CH-006'],
        'chargepad 15w': ['PG-CH-001', 'พัลส์เกียร์ chargepad 15w'],
        'chargepad': ['PG-CH-001', 'PG-CH-002'],
        'soundbar 500': ['KS-SK-001'],
        'soundbar 300': ['KS-SK-002'],
        'soundbar 200': ['KS-SK-003'],
        'soundpillar 300': ['AW-SK-001', 'arcwave soundpillar'],
        'proview 32': ['AW-MN-002'],
        'proview 27 4k': ['AW-MN-001', 'arcwave proview 27'],
        'proview 27': ['AW-MN-001', 'AW-MN-003'],
        'ultraview 34': ['AW-MN-004'],
        'powerbank 30000': ['PG-PB-002', 'พัลส์เกียร์ powerbank 30000'],
        'powerbank 20000': ['PG-PB-001'],
        'powerbank 10000': ['PG-PB-003'],
        'dock pro': ['JC-HB-003', 'จุดเชื่อม dock pro'],
        'hub 7-in-1': ['JC-HB-001', 'จุดเชื่อม hub 7-in-1'],
        'hub 5-in-1': ['JC-HB-002'],
        'usb-c adapter': ['JC-HB-004'],
        'จุดเชื่อม': ['JC-HB', 'JC-CH', 'JC-CS'],
        'พัลส์เกียร์': ['PG-CH', 'PG-PB', 'PG-AC'],
        'arcwave': ['AW-MN', 'AW-SK'],
        # Smart Home
        'smart hub': ['SH-HB-001'],
        'smart bulb': ['SH-LT-001', 'SH-LT-002'],
        'smart plug': ['SH-PW-001'],
        'smart switch': ['SH-PW-002'],
        # Camera
        'actioncam pro': ['AC-CM-001'],
        'actioncam lite': ['AC-CM-002'],
    }
    
    def __init__(self, kb_path: str):
        self.kb_path = Path(kb_path)
        self.documents = {}
        self.doc_metadata = {}
        self.tokenizer = ThaiTokenizer()
        self.df = defaultdict(int)
        self.tf = {}
        self.doc_count = 0
        self._load_knowledge_base()
        self._build_index()
    
    def _load_knowledge_base(self):
        """Load all markdown files from knowledge base."""
        for folder in ['products', 'policies', 'store_info']:
            folder_path = self.kb_path / folder
            if folder_path.exists():
                for file_path in folder_path.glob('*.md'):
                    doc_id = f"{folder}/{file_path.stem}"
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.documents[doc_id] = content
                    self.doc_metadata[doc_id] = {
                        'folder': folder,
                        'filename': file_path.name,
                        'path': str(file_path),
                        'sku': file_path.stem.split('_')[0] if '_' in file_path.stem else ''
                    }
        self.doc_count = len(self.documents)
        print(f"Loaded {self.doc_count} documents from knowledge base")
    
    def _build_index(self):
        """Build TF-IDF index."""
        for doc_id, content in self.documents.items():
            tokens = self.tokenizer.tokenize(content)
            tf_dict = defaultdict(int)
            for token in tokens:
                tf_dict[token] += 1
            self.tf[doc_id] = tf_dict
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] += 1
    
    def _calculate_tfidf_score(self, query_tokens: List[str], doc_id: str) -> float:
        """Calculate TF-IDF similarity score."""
        score = 0.0
        doc_tf = self.tf[doc_id]
        for token in query_tokens:
            if token in doc_tf:
                tf = 1 + math.log(doc_tf[token]) if doc_tf[token] > 0 else 0
                idf = math.log(self.doc_count / (self.df[token] + 1)) if token in self.df else 0
                score += tf * idf
        return score
    
    def _find_product_matches(self, text: str) -> List[str]:
        """Find product aliases in text and return matching doc patterns."""
        text_lower = text.lower()
        matches = []
        for alias, patterns in self.PRODUCT_ALIASES.items():
            if alias in text_lower:
                matches.extend(patterns)
        return matches
    
    def search(self, query: str, top_k: int = 8) -> List[Tuple[str, str, float]]:
        """Search knowledge base for relevant documents."""
        query_tokens = self.tokenizer.tokenize(query)
        product_matches = self._find_product_matches(query)
        
        scores = []
        for doc_id, content in self.documents.items():
            score = self._calculate_tfidf_score(query_tokens, doc_id)
            
            sku = self.doc_metadata[doc_id].get('sku', '')
            for pm in product_matches:
                if pm.upper() in sku.upper() or pm.lower() in content.lower():
                    score *= 3.0
                    break
            
            folder = self.doc_metadata[doc_id]['folder']
            query_lower = query.lower()
            
            if folder == 'policies':
                policy_kw = ['คืน', 'ประกัน', 'ส่ง', 'ยกเลิก', 'จ่าย', 'ชำระ', 'สมาชิก', 'point', 'แต้ม', 
                             'on-site', 'warranty', 'care+', 'เคลม', 'บริการหลังการขาย', 'เปลี่ยน', 
                             'return', 'refund', 'shipping', 'membership', 'คุ้มครอง', 'ความเสียหาย',
                             'น้ำเข้า', 'อุบัติเหตุ', 'สมัครสมาชิก', 'แลกคะแนน', 'ฟรีค่าจัดส่ง']
                if any(kw in query_lower for kw in policy_kw):
                    score *= 2.0
            
            if folder == 'store_info':
                store_kw = ['ร้าน', 'สาขา', 'บริการ', 'สมัคร', 'เทิร์น', 'trade', 'crypto', 'bitcoin', 
                            'สั่ง', 'ชิ้น', 'รายการ', 'จัดส่ง', 'ต่างประเทศ', 'เครดิต', 'บัตรเครดิต',
                            'ผ่อน', 'installment', 'ติดต่อ', 'line', 'facebook', 'รับประกัน',
                            'faq', 'คำถาม', 'วิธีสั่ง', 'payment', 'cryptocurrency']
                if any(kw in query_lower for kw in store_kw):
                    score *= 2.0
            
            if score > 0:
                scores.append((doc_id, content, score))
        
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]
    
    def get_relevant_context(self, question: str, choices: List[str], max_chars: int = 15000) -> str:
        """Get relevant context for a question with its choices."""
        search_parts = [question]
        for c in choices[:8]:
            search_parts.append(c)
        search_query = " ".join(search_parts)
        
        results = self.search(search_query, top_k=8)
        
        if not results:
            return "ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล"
        
        context_parts = []
        total_len = 0
        
        for doc_id, content, score in results:
            if total_len + len(content) > max_chars:
                content = self._extract_key_sections(content)
            
            if total_len + len(content) < max_chars:
                folder = self.doc_metadata[doc_id]['folder']
                context_parts.append(f"=== [{folder}] {doc_id} ===\n{content}")
                total_len += len(content)
        
        return "\n\n".join(context_parts)
    
    def _extract_key_sections(self, content: str) -> str:
        """Extract key sections from document."""
        lines = content.split('\n')
        result = []
        include = True
        section_count = 0
        
        for line in lines:
            if line.startswith('## '):
                section_count += 1
                include = section_count <= 6
            if include:
                result.append(line)
        
        return '\n'.join(result)


_kb_instance = None

def get_knowledge_base(kb_path: str = None) -> KnowledgeBase:
    """Get or create knowledge base instance."""
    global _kb_instance
    if _kb_instance is None:
        if kb_path is None:
            kb_path = "data/knowledge_base"
        _kb_instance = KnowledgeBase(kb_path)
    return _kb_instance
