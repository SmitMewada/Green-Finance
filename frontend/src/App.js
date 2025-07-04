import React, { useState } from 'react';

function App() {
  const [form, setForm] = useState({ name: '', amount: '', description: '' });
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:8000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: form.name,
          amount: parseFloat(form.amount),
          description: form.description
        })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '400px', margin: '0 auto' }}>
      <h1>Green Finance Classifier</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Name"
          value={form.name}
          onChange={(e) => setForm({...form, name: e.target.value})}
          required
          style={{ width: '100%', margin: '5px 0', padding: '8px' }}
        />
        <input
          type="number"
          placeholder="Amount"
          value={form.amount}
          onChange={(e) => setForm({...form, amount: e.target.value})}
          required
          style={{ width: '100%', margin: '5px 0', padding: '8px' }}
        />
        <input
          type="text"
          placeholder="Description"
          value={form.description}
          onChange={(e) => setForm({...form, description: e.target.value})}
          required
          style={{ width: '100%', margin: '5px 0', padding: '8px' }}
        />
        <button type="submit" style={{ width: '100%', margin: '5px 0', padding: '8px' }}>
          Classify
        </button>
      </form>
      {result && (
        <div style={{ marginTop: '20px', padding: '10px', border: '1px solid #ccc' }}>
          <h3>Result:</h3>
          <p><strong>Category:</strong> {result.category}</p>
          <p><strong>Confidence:</strong> {result.confidence}</p>
          <p><strong>ID:</strong> {result.transaction_id}</p>
        </div>
      )}
    </div>
  );
}

export default App;
