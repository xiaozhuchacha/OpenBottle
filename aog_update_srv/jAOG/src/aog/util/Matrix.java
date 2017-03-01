package aog.util;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 * @author Kewei Tu
 * 
 * @param <RH>
 *            the type of row headers
 * @param <CH>
 *            the type of column headers
 */
public class Matrix<RH, CH, CELL> implements Iterable<Triple<RH, CH, CELL>> {
	protected Map<RH, Map<CH, CELL>> rows = new HashMap<RH, Map<CH, CELL>>();
	protected Map<CH, Map<RH, CELL>> cols = new HashMap<CH, Map<RH, CELL>>();

	protected void addRow(RH s) {
		rows.put(s, new HashMap<CH, CELL>());
	}

	protected void addCol(CH s) {
		cols.put(s, new HashMap<RH, CELL>());
	}

	public void delRow(RH s) {
		Map<CH, CELL> m = rows.remove(s);
		if (m != null)
			for (CH c : m.keySet()) {
				Map<RH, CELL> m2 = cols.get(c);
				m2.remove(s);
				if (m2.isEmpty())
					cols.remove(c);
			}
	}

	public void delCol(CH s) {
		Map<RH, CELL> m = cols.remove(s);
		if (m != null)
			for (RH r : m.keySet()) {
				Map<CH, CELL> m2 = rows.get(r);
				m2.remove(s);
				if (m2.isEmpty())
					rows.remove(r);
			}
	}

	public void delCell(RH r, CH c) {
		Map<CH, CELL> row = rows.get(r);
		Map<RH, CELL> col = cols.get(c);
		if (row != null && col != null) {
			row.remove(c);
			col.remove(r);
			if (row.isEmpty())
				delRow(r);
			if (col.isEmpty())
				delCol(c);
		}
	}

	public CELL getCell(RH row, CH col) {
		Map<CH, CELL> r = rows.get(row);
		if (r != null)
			return r.get(col);
		return null;
	}

	public Set<RH> getRowHeaders() {
		return rows.keySet();
	}

	public Set<CH> getColHeaders() {
		return cols.keySet();
	}

	public void setCell(RH r, CH c, CELL cell) {
		if (!rows.containsKey(r))
			addRow(r);
		if (!cols.containsKey(c))
			addCol(c);
		rows.get(r).put(c, cell);
		cols.get(c).put(r, cell);
	}

	public boolean isEmpty() {
		return rows.isEmpty();
	}

	/*
	 * For debugging: display the matrix if small
	 */
	@Override
	public String toString() {
		String re = "size: " + rows.size() + "*" + cols.size() + "\n";

		final int sizelimit = 50;
		ArrayList<CH> cs = new ArrayList<CH>(sizelimit);
		int count = 0;
		for (CH c : cols.keySet()) {
			cs.add(c);
			count++;
			if (count == sizelimit)
				break;
		}
		for (Object ch : cs)
			re += "\t" + ch;
		if (cols.size() > sizelimit)
			re += "\t(" + (cols.size() - sizelimit) + " more cols omitted)";
		count = 0;
		for (Map.Entry<RH, Map<CH, CELL>> r : rows.entrySet()) {
			re += "\n" + r.getKey();
			for (Object ch : cs) {
				re += "\t";
				CELL c = r.getValue().get(ch);
				if (c != null)
					re += cellToString(c);
			}
			count++;
			if (count == sizelimit)
				break;
		}
		if (rows.size() > sizelimit)
			re += "\n(" + (rows.size() - sizelimit) + " more rows omitted)";

		return re;
	}

	protected String cellToString(CELL c) {
		return c.toString();
	}

	public void printToFile(String filename) {
		try {
			FileWriter fw = new FileWriter(filename);

			fw.write("size: " + rows.size() + "*" + cols.size() + "\n");
			ArrayList<CH> cs = new ArrayList<CH>(cols.keySet());
			for (Object ch : cs)
				fw.write("\t" + ch);
			for (Map.Entry<RH, Map<CH, CELL>> r : rows.entrySet()) {
				fw.write("\n" + r.getKey());
				for (Object ch : cs) {
					fw.write("\t");
					CELL c = r.getValue().get(ch);
					if (c != null)
						fw.write(cellToString(c));
				}
			}

			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public String getStatistics() {
		return rows.size() + " rows, " + cols.size() + " columns, "
				+ getCellNumber() + " non-zero elements.";
	}

	public int getCellNumber() {
		int n = 0;
		for (Map<CH, CELL> col : rows.values()) {
			n += col.size();
		}
		return n;
	}

	public int getCellNumberInRow(RH r) {
		return rows.get(r).size();
	}

	public int getCellNumberInCol(CH c) {
		return cols.get(c).size();
	}

	@Override
	public Iterator<Triple<RH, CH, CELL>> iterator() {
		return new MatrixIterator();
	}

	public class MatrixIterator implements Iterator<Triple<RH, CH, CELL>> {
		public MatrixIterator() {
			if (!isEmpty()) {
				riter = rows.entrySet().iterator();
				Entry<RH, Map<CH, CELL>> e = riter.next();
				curRH = e.getKey();
				citer = e.getValue().entrySet().iterator();
			}
		}

		protected Iterator<Entry<RH, Map<CH, CELL>>> riter;
		protected RH curRH;
		protected Iterator<Entry<CH, CELL>> citer;

		@Override
		public boolean hasNext() {
			if (isEmpty())
				return false;

			if (citer.hasNext())
				return true;
			else
				return riter.hasNext();
		}

		@Override
		public Triple<RH, CH, CELL> next() {
			if (!hasNext())
				throw new NoSuchElementException();

			if (!citer.hasNext()) {
				Entry<RH, Map<CH, CELL>> e = riter.next();
				curRH = e.getKey();
				citer = e.getValue().entrySet().iterator();
			}
			Entry<CH, CELL> e = citer.next();
			return new Triple<>(curRH, e.getKey(), e.getValue());
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}
	};

	/**
	 * Whether the two matrices have the same sets of rows and columns
	 * 
	 * @param obj
	 * @return
	 */
	public boolean hasEqualHeaders(Matrix<?, ?, ?> bc2) {
		return rows.keySet().equals(bc2.rows.keySet())
				&& cols.keySet().equals(bc2.cols.keySet());
	}

	/**
	 * Whether the two matrices have the same sets of rows and columns after one
	 * of the matrices is transposed.
	 * 
	 * @param obj
	 * @return
	 */
	public boolean hasEqualHeadersAfterTransposition(Matrix<?, ?, ?> bc2) {
		return cols.keySet().equals(bc2.rows.keySet())
				&& rows.keySet().equals(bc2.cols.keySet());
	}
}
