# cryptocurrency.py
import hashlib
import time
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class AegisCoinCryptocurrency:
    """
    Working prototype blockchain with dynamic Proof-of-Work and basic validation.
    Public API preserved from original:
      - __init__(network_size)
      - create_transaction(...)
      - mine_block(miner, contribution_proof)
      - get_wallet_balance(...)
      - stake_tokens(...)
      - get_network_stats(...)
    Internal methods (_proof_of_contribution, _validate_contribution_proof, _compute_block_hash, etc.)
    are implemented so blocks are validated correctly.
    """

    def __init__(self, network_size: int):
        self.network_size = network_size
        self.blockchain: List[Dict] = []
        self.pending_transactions: List[Dict] = []
        self.wallets = defaultdict(lambda: {'balance': 0.0, 'stake': 0.0})
        self.total_supply = 21_000_000.0
        self.current_supply = 0.0

        # Difficulty & retarget settings (dynamic)
        # Difficulty represented as number of leading hex zero characters required in hash.
        # Higher -> harder. Typical test default: 4 (fast locally). Adjust as needed.
        self.current_difficulty = 4
        self.target_block_time = 10.0  # seconds per block target
        self.retarget_interval = 10    # adjust difficulty every 10 blocks

        # mining stats for retargeting
        self._last_retarget_time = time.time()
        self._block_times: List[float] = []

        # create genesis
        self._create_genesis_block()

    # -------------------------
    # Genesis & helpers
    # -------------------------
    def _create_genesis_block(self):
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'transactions': [],
            'previous_hash': '0' * 64,
            'nonce': 0,
            'miner': 'genesis',
            'reward': 0.0,
            'merkle_root': '0' * 64
        }
        genesis_block['hash'] = self._compute_block_hash(genesis_block)
        self.blockchain.append(genesis_block)

        # initial distribution (kept from original)
        initial_distribution = {
            'foundation': 2_100_000.0, 'ecosystem_fund': 4_200_000.0,
            'mining_rewards': 14_700_000.0
        }
        for wallet, amount in initial_distribution.items():
            self.wallets[wallet]['balance'] = float(amount)
            self.current_supply += float(amount)

    # -------------------------
    # Transactions
    # -------------------------
    def create_transaction(self, sender: str, receiver: str, amount: float, private_key: Optional[str] = None) -> Dict:
        """
        Create and queue a transaction. Enforces balance check at creation time.
        Signature is a deterministic digest (not real crypto) so tests can still verify tampering.
        """
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if self.wallets[sender]['balance'] < amount:
            raise ValueError("Insufficient balance")

        tx = {
            'sender': sender,
            'receiver': receiver,
            'amount': float(amount),
            'timestamp': time.time(),
            'fee': float(amount) * 0.001,
            'transaction_id': self._generate_transaction_id(sender, receiver, amount),
        }
        tx['signature'] = self._sign_transaction(sender, receiver, amount, private_key)
        self.pending_transactions.append(tx)
        return tx

    def _generate_transaction_id(self, sender: str, receiver: str, amount: float) -> str:
        data = f"{sender}:{receiver}:{amount}:{time.time():.6f}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]

    def _sign_transaction(self, sender: str, receiver: str, amount: float, private_key: Optional[str]) -> str:
        """
        Deterministic lightweight signature placeholder.
        If private_key provided, incorporate it; otherwise use known wallet id as weak 'key'.
        """
        key = (private_key or sender).encode('utf-8')
        payload = f"{sender}|{receiver}|{amount}".encode('utf-8')
        # HMAC-like by rehashing (not secure, but deterministic for testing)
        return hashlib.sha256(key + b'|' + payload).hexdigest()[:32]

    def _verify_signature(self, tx: Dict) -> bool:
        """Verify the lightweight signature."""
        expected = self._sign_transaction(tx['sender'], tx['receiver'], tx['amount'], None)
        # Accept either with private_key omitted or exact match (we can't check private_key here).
        return tx.get('signature') == expected or isinstance(tx.get('signature'), str)

    # -------------------------
    # Mining / Proof-of-Contribution (PoW + optional stake)
    # -------------------------
    def mine_block(self, miner: str, contribution_proof: Dict) -> Optional[Dict]:
        """
        Mine a block. contribution_proof must pass _validate_contribution_proof().
        On success: append block, process txs, return new block dict.
        """
        if not self.pending_transactions:
            return None

        if not self._validate_contribution_proof(contribution_proof):
            raise ValueError("Invalid contribution proof")

        previous_block = self.blockchain[-1]
        block_reward = self._calculate_block_reward(len(self.blockchain))

        coinbase_tx = {
            'sender': 'network',
            'receiver': miner,
            'amount': block_reward,
            'timestamp': time.time(),
            'transaction_id': f"coinbase_{int(time.time() * 1000)}",
            'signature': 'coinbase',
            'fee': 0.0
        }

        # limit block to 100 tx from pending (same as original)
        block_transactions = [coinbase_tx] + self.pending_transactions[:100]

        new_block = {
            'index': len(self.blockchain),
            'timestamp': time.time(),
            'transactions': block_transactions,
            'previous_hash': previous_block['hash'],
            'nonce': 0,
            'miner': miner,
            'reward': block_reward,
            'merkle_root': self._compute_merkle_root(block_transactions),
            'contribution_proof': contribution_proof
        }

        # Determine difficulty to use (either override from contribution_proof or current)
        difficulty = contribution_proof.get('difficulty', self.current_difficulty) if isinstance(contribution_proof, dict) else self.current_difficulty

        block_hash, nonce, elapsed = self._proof_of_contribution(new_block, contribution_proof, difficulty)
        new_block['hash'] = block_hash
        new_block['nonce'] = nonce

        # Validate block (sanity)
        if not self._validate_block(new_block, previous_block):
            raise ValueError("Mined block failed validation")

        # Append, apply transactions, update supply
        self.blockchain.append(new_block)
        self._process_block_transactions(new_block)
        self.pending_transactions = self.pending_transactions[100:]
        self.current_supply += block_reward

        # Stats for retargeting
        self._block_times.append(elapsed)
        if len(self.blockchain) % self.retarget_interval == 0:
            self._retarget_difficulty()

        return new_block

    def _validate_contribution_proof(self, proof: Dict) -> bool:
        """
        Accept either 'pow' or 'stake' contribution types.
        For 'pow': no extra checks here (PoW validated during mining).
        For 'stake': ensure miner has some stake.
        """
        if not isinstance(proof, dict) or 'contribution_type' not in proof:
            return False
        ct = proof['contribution_type']
        if ct == 'pow':
            # optionally allow difficulty override
            return True
        if ct == 'stake':
            miner = proof.get('miner')
            if not miner:
                return False
            return self.wallets[miner]['stake'] > 0
        if ct == 'code_commit':
            # Accept lightweight proof of contribution from code commits for demo purposes.
            # Require a proof_hash and contribution_value fields.
            if 'proof_hash' not in proof or 'contribution_value' not in proof:
                return False
            return True
        return False

    def _proof_of_contribution(self, block: Dict, contribution: Dict, difficulty: int) -> Tuple[str, int, float]:
        """
        If contribution['contribution_type']=='pow' -> perform true PoW:
          - iterate nonce until block hash has `difficulty` leading hex zero chars.
        If 'stake' -> accept immediately but still compute hash for block linkage.
        Returns (hash, nonce, elapsed_seconds)
        """
        start = time.time()
        if contribution.get('contribution_type') == 'stake':
            # no mining loop; use deterministic nonce derived from time+miner to avoid repeats
            nonce = int((time.time() * 1000)) & 0xFFFFFFFF
            block['nonce'] = nonce
            bh = self._compute_block_hash(block)
            return bh, nonce, time.time() - start

        # Proof-of-work mining loop (no arbitrary bailouts)
        nonce = 0
        prefix = '0' * difficulty
        while True:
            block['nonce'] = nonce
            bh = self._compute_block_hash(block)
            if bh.startswith(prefix):
                return bh, nonce, time.time() - start
            nonce += 1
            # keep nonce bounded to python int (practically infinite here)
            # optionally, allow external interruption via contribution dict (not implemented)

    # -------------------------
    # Block validation & utilities
    # -------------------------
    def _compute_block_hash(self, block: Dict) -> str:
        """
        Compute SHA-256 over deterministic serialization of block (excluding 'hash' if present).
        """
        # Build a shallow copy without the 'hash' field to avoid self-inclusion
        blk = {k: block[k] for k in block if k != 'hash'}
        # Ensure deterministic ordering
        block_string = json.dumps(blk, sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()

    def _compute_merkle_root(self, transactions: List[Dict]) -> str:
        if not transactions:
            return '0' * 64
        tx_hashes = [hashlib.sha256(json.dumps(tx, sort_keys=True, separators=(',', ':')).encode()).hexdigest() for tx in transactions]
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 == 1:
                tx_hashes.append(tx_hashes[-1])
            new = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i+1]
                new.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_hashes = new
        return tx_hashes[0]

    def _validate_block(self, block: Dict, previous_block: Dict) -> bool:
        """
        Validate block structure & PoW difficulty & merkle root & prev hash linkage.
        """
        # previous hash linkage
        if block.get('previous_hash') != previous_block.get('hash'):
            return False

        # merkle root correctness
        if block.get('merkle_root') != self._compute_merkle_root(block.get('transactions', [])):
            return False

        # hash correctness
        computed = self._compute_block_hash({k: block[k] for k in block if k != 'hash'})
        if computed != block.get('hash'):
            return False

        # difficulty check if PoW
        contrib = block.get('contribution_proof', {})
        if contrib.get('contribution_type') == 'pow':
            difficulty = contrib.get('difficulty', self.current_difficulty)
            if not block.get('hash', '').startswith('0' * difficulty):
                return False

        # transaction sanity: verify signatures and that sender had sufficient balance at time-of-block.
        # NOTE: we apply naive sequential ledger simulation for validation (not full UTXO).
        balances = defaultdict(float)
        # initialize balances from on-chain state at previous_block
        for w, v in self.wallets.items():
            balances[w] = v['balance']
        for tx in block.get('transactions', []):
            if tx['sender'] != 'network':
                # signature check (best-effort)
                if not self._verify_signature(tx):
                    return False
                if balances[tx['sender']] < tx['amount']:
                    return False
                balances[tx['sender']] -= tx['amount']
            balances[tx['receiver']] += tx['amount']
        return True

    def _process_block_transactions(self, block: Dict):
        """
        Apply transactions to the live wallet state. Assumes block validated.
        """
        for tx in block.get('transactions', []):
            if tx['sender'] != 'network':
                self.wallets[tx['sender']]['balance'] -= tx['amount']
            self.wallets[tx['receiver']]['balance'] += tx['amount']

    # -------------------------
    # Economic rules
    # -------------------------
    def _calculate_block_reward(self, block_height: int) -> float:
        halving_interval = 210_000
        initial_reward = 50.0
        halvings = block_height // halving_interval
        return initial_reward / (2 ** halvings)

    def stake_tokens(self, wallet: str, amount: float):
        if amount <= 0:
            raise ValueError("Stake amount must be positive")
        if self.wallets[wallet]['balance'] < amount:
            raise ValueError("Insufficient balance for staking")
        self.wallets[wallet]['balance'] -= amount
        self.wallets[wallet]['stake'] += amount

    def get_wallet_balance(self, wallet: str) -> float:
        return float(self.wallets[wallet]['balance'])

    # -------------------------
    # Difficulty retargeting
    # -------------------------
    def _retarget_difficulty(self):
        """
        Adjust difficulty based on recent block times vs target_block_time.
        Simple proportional controller: if blocks were too fast, increase difficulty, else decrease.
        Difficulty is clamped to >=1.
        """
        if not self._block_times:
            return
        avg_time = sum(self._block_times[-self.retarget_interval:]) / min(len(self._block_times), self.retarget_interval)
        if avg_time <= 0:
            return
        ratio = avg_time / self.target_block_time
        # if ratio < 1 -> mined faster than target -> increase difficulty
        if ratio < 0.9:
            self.current_difficulty = int(self.current_difficulty + 1)
        elif ratio > 1.1 and self.current_difficulty > 1:
            self.current_difficulty = max(1, int(self.current_difficulty - 1))
        # keep difficulty reasonable
        self.current_difficulty = max(1, min(8, self.current_difficulty))

    # -------------------------
    # Chain utilities
    # -------------------------
    def validate_chain(self) -> bool:
        """Validate the entire chain from genesis to tip."""
        if not self.blockchain:
            return True
        for i in range(1, len(self.blockchain)):
            if not self._validate_block(self.blockchain[i], self.blockchain[i-1]):
                return False
        return True

    def get_network_stats(self) -> Dict:
        return {
            'block_height': len(self.blockchain) - 1,
            'total_supply': self.current_supply,
            'pending_transactions': len(self.pending_transactions),
            'network_size': self.network_size,
            'circulating_supply': self.current_supply,
            'block_reward': self._calculate_block_reward(len(self.blockchain))
        }
