"""
Kiwoom Securities Open API Wrapper

WARNING: This module connects to real trading account.
ALWAYS use paper trading account for testing.
"""

import sys
from typing import Dict, List, Optional, Callable
from PyQt5.QAxContainer import QAxWidget
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from src.utils.logger import LoggerMixin


class KiwoomAPI(LoggerMixin):
    """
    Kiwoom Open API wrapper for Python.

    Features:
    - Login and authentication
    - Real-time market data
    - Order execution (buy/sell)
    - Account information
    - Balance and holdings

    Requirements:
    - Windows OS
    - Kiwoom OpenAPI+ installed
    - PyQt5

    Example:
    --------
    >>> api = KiwoomAPI()
    >>> api.comm_connect()
    >>> # Wait for login
    >>> account = api.get_login_info("ACCNO")
    >>> api.send_order("buy_order", "0101", account[0], 1, "005930", 10, 0, "00", "")
    """

    def __init__(self):
        """Initialize Kiwoom API."""
        super().__init__()

        # QApplication instance (필수)
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)

        # OCX 객체 생성
        self.ocx = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")

        # Event loop for synchronous calls
        self.login_event_loop = None
        self.order_event_loop = None

        # 접수/체결 데이터 저장
        self.order_data = {}
        self.chejan_data = {}

        # TR 요청 데이터 저장
        self.tr_data = {}
        self.tr_event_loop = None

        # 실시간 데이터 콜백
        self.realtime_callbacks = {}

        # 연결 이벤트
        self._connect_signals()

        self.logger.info("Kiwoom API initialized")

    def _connect_signals(self):
        """Connect OCX events to handlers."""
        # 로그인 이벤트
        self.ocx.OnEventConnect.connect(self._on_event_connect)

        # TR 수신 이벤트
        self.ocx.OnReceiveTrData.connect(self._on_receive_tr_data)

        # 실시간 데이터 수신
        self.ocx.OnReceiveRealData.connect(self._on_receive_real_data)

        # 주문 체결 데이터 수신
        self.ocx.OnReceiveChejanData.connect(self._on_receive_chejan_data)

        # 메시지 수신
        self.ocx.OnReceiveMsg.connect(self._on_receive_msg)

    # ========== Login ==========

    def comm_connect(self, timeout: int = 30000) -> int:
        """
        로그인 윈도우 실행.

        Parameters
        ----------
        timeout : int
            Login timeout in milliseconds

        Returns
        -------
        int
            0: success, other: failure
        """
        self.logger.info("Attempting to connect to Kiwoom server...")

        self.login_event_loop = QEventLoop()

        # Timeout timer
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self.login_event_loop.exit(-1))
        timer.start(timeout)

        # Request login
        ret = self.ocx.dynamicCall("CommConnect()")

        if ret == 0:
            self.logger.info("Login request sent, waiting for response...")
            result = self.login_event_loop.exec_()

            if result == 0:
                self.logger.info("Successfully logged in")
                return 0
            else:
                self.logger.error("Login timeout")
                return -1
        else:
            self.logger.error(f"Login request failed: {ret}")
            return ret

    def _on_event_connect(self, err_code: int):
        """로그인 이벤트 핸들러."""
        if err_code == 0:
            self.logger.info("Login successful")
        else:
            self.logger.error(f"Login failed: {err_code}")

        if self.login_event_loop:
            self.login_event_loop.exit(err_code)

    def get_connect_state(self) -> int:
        """
        접속 상태 확인.

        Returns
        -------
        int
            0: disconnected, 1: connected
        """
        return self.ocx.dynamicCall("GetConnectState()")

    def get_login_info(self, tag: str) -> str:
        """
        로그인 정보 조회.

        Parameters
        ----------
        tag : str
            "ACCNO": 계좌번호
            "USER_ID": 사용자ID
            "USER_NAME": 사용자명
            "KEY_BSECGB": 키보드보안 해지 여부
            "FIREW_SECGB": 방화벽 설정 여부

        Returns
        -------
        str
            Requested information
        """
        return self.ocx.dynamicCall("GetLoginInfo(QString)", tag)

    # ========== TR Data Request ==========

    def set_input_value(self, id: str, value: str):
        """SetInputValue."""
        self.ocx.dynamicCall("SetInputValue(QString, QString)", id, value)

    def comm_rq_data(self, rqname: str, trcode: str, prev_next: int, screen_no: str) -> int:
        """
        TR 요청.

        Parameters
        ----------
        rqname : str
            사용자 구분명
        trcode : str
            TR 코드
        prev_next : int
            연속조회 여부 (0: 조회, 2: 연속)
        screen_no : str
            화면번호

        Returns
        -------
        int
            0: success, other: failure
        """
        ret = self.ocx.dynamicCall(
            "CommRqData(QString, QString, int, QString)",
            rqname, trcode, prev_next, screen_no
        )

        if ret == 0:
            self.tr_event_loop = QEventLoop()
            self.tr_event_loop.exec_()

        return ret

    def _on_receive_tr_data(
        self,
        screen_no: str,
        rqname: str,
        trcode: str,
        record_name: str,
        prev_next: str,
        *args
    ):
        """TR 데이터 수신 이벤트."""
        self.logger.debug(f"Received TR data: {rqname} ({trcode})")

        if prev_next == '2':
            self.tr_data['has_next'] = True
        else:
            self.tr_data['has_next'] = False

        if self.tr_event_loop:
            self.tr_event_loop.exit()

    def get_comm_data(
        self,
        trcode: str,
        record_name: str,
        index: int,
        item_name: str
    ) -> str:
        """GetCommData - TR 데이터 추출."""
        return self.ocx.dynamicCall(
            "GetCommData(QString, QString, int, QString)",
            trcode, record_name, index, item_name
        ).strip()

    def get_repeat_cnt(self, trcode: str, record_name: str) -> int:
        """GetRepeatCnt - 멀티데이터 반복 횟수."""
        return self.ocx.dynamicCall(
            "GetRepeatCnt(QString, QString)",
            trcode, record_name
        )

    # ========== Order ==========

    def send_order(
        self,
        rqname: str,
        screen_no: str,
        acc_no: str,
        order_type: int,
        code: str,
        quantity: int,
        price: int,
        hoga_gb: str,
        org_order_no: str = ""
    ) -> int:
        """
        주문 전송.

        Parameters
        ----------
        rqname : str
            사용자 구분명
        screen_no : str
            화면번호
        acc_no : str
            계좌번호
        order_type : int
            주문유형 (1: 신규매수, 2: 신규매도, 3: 매수취소, 4: 매도취소, 5: 매수정정, 6: 매도정정)
        code : str
            종목코드
        quantity : int
            주문수량
        price : int
            주문가격 (0: 시장가)
        hoga_gb : str
            호가구분 ("00": 지정가, "03": 시장가, ...)
        org_order_no : str
            원주문번호 (정정/취소 시)

        Returns
        -------
        int
            0: success, other: failure
        """
        ret = self.ocx.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
            [rqname, screen_no, acc_no, order_type, code, quantity, price, hoga_gb, org_order_no]
        )

        if ret == 0:
            self.logger.info(
                f"Order sent: {code} {'BUY' if order_type == 1 else 'SELL'} "
                f"{quantity} @ {price if price > 0 else 'MARKET'}"
            )
        else:
            self.logger.error(f"Order failed: {ret}")

        return ret

    def _on_receive_chejan_data(self, gubun: str, item_cnt: int, fid_list: str):
        """체결 데이터 수신."""
        if gubun == "0":  # 주문체결
            self.logger.info("Order filled")
            # 체결 정보 저장
            self.chejan_data = {
                'order_no': self.get_chejan_data(9203),
                'code': self.get_chejan_data(9001),
                'order_qty': int(self.get_chejan_data(900)),
                'filled_qty': int(self.get_chejan_data(911)),
                'price': int(self.get_chejan_data(910)),
                'order_status': self.get_chejan_data(913)
            }

        elif gubun == "1":  # 잔고
            self.logger.debug("Balance updated")

    def get_chejan_data(self, fid: int) -> str:
        """GetChejanData."""
        return self.ocx.dynamicCall("GetChejanData(int)", fid).strip()

    def _on_receive_msg(
        self,
        screen_no: str,
        rqname: str,
        trcode: str,
        msg: str
    ):
        """메시지 수신."""
        self.logger.info(f"Message: {msg}")

    # ========== Real-time Data ==========

    def set_real_reg(
        self,
        screen_no: str,
        code_list: str,
        fid_list: str,
        opt_type: str
    ) -> int:
        """
        실시간 등록.

        Parameters
        ----------
        screen_no : str
            화면번호
        code_list : str
            종목코드 리스트 (세미콜론 구분)
        fid_list : str
            FID 리스트 (세미콜론 구분)
        opt_type : str
            등록구분 ("0": 추가, "1": 삭제)

        Returns
        -------
        int
            Result code
        """
        ret = self.ocx.dynamicCall(
            "SetRealReg(QString, QString, QString, QString)",
            screen_no, code_list, fid_list, opt_type
        )

        if ret == 0:
            self.logger.info(f"Real-time registration: {code_list}")
        else:
            self.logger.error(f"Real-time registration failed: {ret}")

        return ret

    def set_real_remove(self, screen_no: str, code: str):
        """실시간 해제."""
        self.ocx.dynamicCall("SetRealRemove(QString, QString)", screen_no, code)

    def _on_receive_real_data(self, code: str, real_type: str, real_data: str):
        """실시간 데이터 수신."""
        # 등록된 콜백 실행
        if code in self.realtime_callbacks:
            self.realtime_callbacks[code](code, real_type, real_data)

    def get_comm_real_data(self, code: str, fid: int) -> str:
        """GetCommRealData."""
        return self.ocx.dynamicCall("GetCommRealData(QString, int)", code, fid).strip()

    def register_realtime_callback(self, code: str, callback: Callable):
        """실시간 데이터 콜백 등록."""
        self.realtime_callbacks[code] = callback

    # ========== Account Info ==========

    def get_account_list(self) -> List[str]:
        """계좌번호 리스트 조회."""
        accounts = self.get_login_info("ACCNO")
        return accounts.split(';')[:-1] if accounts else []

    def get_balance(self, account: str) -> Dict:
        """
        계좌 잔고 조회.

        Parameters
        ----------
        account : str
            계좌번호

        Returns
        -------
        dict
            Balance information
        """
        self.set_input_value("계좌번호", account)
        self.set_input_value("비밀번호", "")
        self.set_input_value("비밀번호입력매체구분", "00")
        self.set_input_value("조회구분", "1")

        self.comm_rq_data("계좌평가잔고내역요청", "opw00018", 0, "2000")

        # Parse balance data (simplified)
        return self.tr_data

    # ========== Utilities ==========

    def get_code_list_by_market(self, market: str) -> List[str]:
        """
        시장별 종목코드 리스트.

        Parameters
        ----------
        market : str
            "0": 코스피, "10": 코스닥, "8": ETF, ...

        Returns
        -------
        list
            Code list
        """
        codes = self.ocx.dynamicCall("GetCodeListByMarket(QString)", market)
        return codes.split(';')[:-1] if codes else []

    def get_master_code_name(self, code: str) -> str:
        """종목명 조회."""
        return self.ocx.dynamicCall("GetMasterCodeName(QString)", code)

    def disconnect(self):
        """연결 해제."""
        self.ocx.dynamicCall("CommTerminate()")
        self.logger.info("Disconnected from Kiwoom server")
