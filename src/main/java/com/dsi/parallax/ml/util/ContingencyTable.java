/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util;

public class ContingencyTable {

    private final int numRows, numCols;
    private final double[] rowTotals, colTotals;
    private final double[][] entryValues;
    private double total;

    public ContingencyTable(int numRows, int numCols) {
        this.numRows = numRows;
        this.numCols = numCols;
        rowTotals = new double[numRows];
        colTotals = new double[numCols];
        total = 0;
        entryValues = new double[numRows][];
    }

    public void addValue(int row, int column, double value) {

        rowTotals[row] = value;
        colTotals[column] = value;

        entryValues[row][column] = value;
        total += value;
    }

    /**
     * Returns chi-squared probability for a given matrix.
     * 
     * @param matrix
     *            the contigency table
     * @param yates
     *            is Yates' correction to be used?
     * @return the chi-squared probability
     */

    public double chiSquared(boolean yates) {

        int df = (numRows - 1) * (numCols - 1);

        return Stat.chiSquaredProbability(chiVal(yates), df);
    }

    /**
     * Computes chi-squared statistic for a contingency table.
     * 
     * @param matrix
     *            the contigency table
     * @param useYates
     *            is Yates' correction to be used?
     * @return the value of the chi-squared statistic
     */
    public double chiVal(boolean useYates) {

        int df, row, col;
        double expect = 0, chival = 0, n = 0;
        boolean yates = true;

        df = (numRows - 1) * (numCols - 1);
        if ((df > 1) || (!useYates)) {
            yates = false;
        } else if (df <= 0) {
            return 0;
        }
        chival = 0.0;
        for (row = 0; row < numRows; row++) {
            if (MLUtils.floatingPointGreaterThan(rowTotals[row], 0)) {
                for (col = 0; col < numCols; col++) {
                    if (MLUtils.floatingPointGreaterThan(colTotals[col], 0)) {
                        expect = (colTotals[col] * rowTotals[row]) / n;
                        chival += chiCell(entryValues[row][col], expect, yates);
                    }
                }
            }
        }
        return chival;
    }

    /**
     * Computes chi-value for one cell in a contingency table.
     * 
     * @param freq
     *            the observed frequency in the cell
     * @param expected
     *            the expected frequency in the cell
     * @return the chi-value for that cell; 0 if the expected value is too close
     *         to zero
     */
    private static double chiCell(double freq, double expected, boolean yates) {

        // Cell in empty row and column?
        if (MLUtils.floatingPointLessThanOrEquals(expected, 0)) {
            return 0;
        }

        // Compute difference between observed and expected value
        double diff = Math.abs(freq - expected);
        if (yates) {

            // Apply Yates' correction if wanted
            diff -= 0.5;

            // The difference should never be negative
            if (diff < 0) {
                diff = 0;
            }
        }

        // Return chi-value for the cell
        return (diff * diff / expected);
    }

    /**
     * Tests if Cochran's criterion is fullfilled for the given contingency
     * table. Rows and columns with all zeros are not considered relevant.
     * 
     * @param matrix
     *            the contigency table to be tested
     * @return true if contingency table is ok, false if not
     */
    public boolean cochransCriterion() {
        if(total<=0)
            return false;
        
        double expect, smallfreq = 5;
        int smallcount = 0, nonZeroRows = 0, nonZeroColumns = 0;

        for (int row = 0; row < numRows; row++) {
            if (MLUtils.floatingPointGreaterThan(rowTotals[row], 0)) {
                nonZeroRows++;
            }
        }
        for (int col = 0; col < numCols; col++) {
            if (MLUtils.floatingPointGreaterThan(colTotals[col], 0)) {
                nonZeroColumns++;
            }
        }
        for (int row = 0; row < numRows; row++) {
            if (MLUtils.floatingPointGreaterThan(rowTotals[row], 0)) {
                for (int col = 0; col < numCols; col++) {
                    if (MLUtils.floatingPointGreaterThan(colTotals[col], 0)) {
                        expect = (colTotals[col] * rowTotals[row]) / total;
                        if (MLUtils.floatingPointLessThan(expect, smallfreq)) {
                            if (MLUtils.floatingPointLessThan(expect, 1)) {
                                return false;
                            } else {
                                smallcount++;
                                if (smallcount > (nonZeroRows * nonZeroColumns)
                                        / smallfreq) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

    /**
     * Computes Cramer's V for a contingency table.
     * 
     * @param matrix
     *            the contingency table
     * @return Cramer's V
     */
    public double CramersV() {

        int min = numRows < numCols ? numRows -1 : numCols -1;
        
        if ((min == 0) || MLUtils.floatingPointEquals(total, 0))
            return 0;
        return Math.sqrt(chiVal(false) / (total * min));
    }



    private static double lnFunc(double num) {

        // Constant hard coded for efficiency reasons
        if (num < 1e-6) {
            return 0;
        } else {
            return num * Math.log(num);
        }
    }

    public double entropyConditionedOnColumnds() {
        return entropyConditionedOnColumns(entryValues);
    }
    
    /**
     * Computes conditional entropy of the rows given the columns.
     * 
     * @param matrix
     *            the contingency table
     * @return the conditional entropy of the rows given the columns
     */
    public static double entropyConditionedOnColumns(double[][] matrix) {

        double returnValue = 0, sumForColumn, total = 0;

        for (int j = 0; j < matrix[0].length; j++) {
            sumForColumn = 0;
            for (int i = 0; i < matrix.length; i++) {
                returnValue = returnValue + lnFunc(matrix[i][j]);
                sumForColumn += matrix[i][j];
            }
            returnValue = returnValue - lnFunc(sumForColumn);
            total += sumForColumn;
        }
        if (MLUtils.floatingPointEquals(total, 0)) {
            return 0;
        }
        return -returnValue / (total * MLUtils.LOG2);
    }

    public double entropyConditionedOnRows() {
        return entropyConditionedOnRows(entryValues);
    }
    
    /**
     * Computes conditional entropy of the columns given the rows.
     * 
     * @param matrix
     *            the contingency table
     * @return the conditional entropy of the columns given the rows
     */
    public static double entropyConditionedOnRows(double[][] matrix) {

        double returnValue = 0, sumForRow, total = 0;

        for (int i = 0; i < matrix.length; i++) {
            sumForRow = 0;
            for (int j = 0; j < matrix[0].length; j++) {
                returnValue = returnValue + lnFunc(matrix[i][j]);
                sumForRow += matrix[i][j];
            }
            returnValue = returnValue - lnFunc(sumForRow);
            total += sumForRow;
        }
        if (MLUtils.floatingPointEquals(total, 0)) {
            return 0;
        }
        return -returnValue / (total * MLUtils.LOG2);
    }

    /**
     * Computes conditional entropy of the columns given the rows of the test
     * matrix with respect to the train matrix. Uses a Laplace prior. Does NOT
     * normalize the entropy.
     * 
     * @param train
     *            the train matrix
     * @param test
     *            the test matrix
     * @param numClasses
     *            the number of symbols for Laplace
     * @return the entropy
     */
    public static double entropyConditionedOnRows(double[][] train,
            double[][] test, double numClasses) {

        double returnValue = 0, trainSumForRow, testSumForRow, testSum = 0;

        for (int i = 0; i < test.length; i++) {
            trainSumForRow = 0;
            testSumForRow = 0;
            for (int j = 0; j < test[0].length; j++) {
                returnValue -= test[i][j] * Math.log(train[i][j] + 1);
                trainSumForRow += train[i][j];
                testSumForRow += test[i][j];
            }
            testSum = testSumForRow;
            returnValue += testSumForRow
                    * Math.log(trainSumForRow + numClasses);
        }
        return returnValue / (testSum * MLUtils.LOG2);
    }

    public double entropyOverRows() {
        return entropyOverRows(entryValues);
    }
    
    /**
     * Computes the rows' entropy for the given contingency table.
     * 
     * @param matrix
     *            the contingency table
     * @return the rows' entropy
     */
    public static double entropyOverRows(double[][] matrix) {

        double returnValue = 0, sumForRow, total = 0;

        for (int i = 0; i < matrix.length; i++) {
            sumForRow = 0;
            for (int j = 0; j < matrix[0].length; j++) {
                sumForRow += matrix[i][j];
            }
            returnValue = returnValue - lnFunc(sumForRow);
            total += sumForRow;
        }
        if (MLUtils.floatingPointEquals(total, 0)) {
            return 0;
        }
        return (returnValue + lnFunc(total)) / (total * MLUtils.LOG2);
    }

    public double entropyOverColumns() {
        return entropyOverColumns(entryValues);
    }
    
    /**
     * Computes the columns' entropy for the given contingency table.
     * 
     * @param matrix
     *            the contingency table
     * @return the columns' entropy
     */
    public static double entropyOverColumns(double[][] matrix) {

        double returnValue = 0, sumForColumn, total = 0;

        for (int j = 0; j < matrix[0].length; j++) {
            sumForColumn = 0;
            for (int i = 0; i < matrix.length; i++) {
                sumForColumn += matrix[i][j];
            }
            returnValue = returnValue - lnFunc(sumForColumn);
            total += sumForColumn;
        }
        if (MLUtils.floatingPointEquals(total, 0)) {
            return 0;
        }
        return (returnValue + lnFunc(total)) / (total * MLUtils.LOG2);
    }

    public double gainRatio() {
        return gainRatio(entryValues);
    }
    
    /**
     * Computes gain ratio for contingency table (split on rows). Returns
     * Double.MAX_VALUE if the split entropy is 0.
     * 
     * @param matrix
     *            the contingency table
     * @return the gain ratio
     */
    public static double gainRatio(double[][] matrix) {

        double preSplit = 0, postSplit = 0, splitEnt = 0, sumForRow, sumForColumn, total = 0, infoGain;

        // Compute entropy before split
        for (int i = 0; i < matrix[0].length; i++) {
            sumForColumn = 0;
            for (int j = 0; j < matrix.length; j++)
                sumForColumn += matrix[j][i];
            preSplit += lnFunc(sumForColumn);
            total += sumForColumn;
        }
        preSplit -= lnFunc(total);

        // Compute entropy after split and split entropy
        for (int i = 0; i < matrix.length; i++) {
            sumForRow = 0;
            for (int j = 0; j < matrix[0].length; j++) {
                postSplit += lnFunc(matrix[i][j]);
                sumForRow += matrix[i][j];
            }
            splitEnt += lnFunc(sumForRow);
        }
        postSplit -= splitEnt;
        splitEnt -= lnFunc(total);

        infoGain = preSplit - postSplit;
        if (MLUtils.floatingPointEquals(splitEnt, 0))
            return 0;
        return infoGain / splitEnt;
    }

    public double log2MultipleHypergeometric() {
        return log2MultipleHypergeometric(entryValues);
    }
    
    /**
     * Returns negative base 2 logarithm of multiple hypergeometric probability
     * for a contingency table.
     * 
     * @param matrix
     *            the contingency table
     * @return the log of the hypergeometric probability of the contingency
     *         table
     */
    public static double log2MultipleHypergeometric(double[][] matrix) {

        double sum = 0, sumForRow, sumForColumn, total = 0;

        for (int i = 0; i < matrix.length; i++) {
            sumForRow = 0;
            for (int j = 0; j < matrix[i].length; j++) {
                sumForRow += matrix[i][j];
            }
            sum += Stat.logNfact(sumForRow);
            total += sumForRow;
        }
        for (int j = 0; j < matrix[0].length; j++) {
            sumForColumn = 0;
            for (int i = 0; i < matrix.length; i++) {
                sumForColumn += matrix[i][j];
            }
            sum += Stat.logNfact(sumForColumn);
        }
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                sum -= Stat.logNfact(matrix[i][j]);
            }
        }
        sum -= Stat.logNfact(total);
        return -sum / MLUtils.LOG2;
    }

    /**
     * Reduces a matrix by deleting all zero rows and columns.
     * 
     * @param matrix
     *            the matrix to be reduced
     * @return the matrix with all zero rows and columns deleted
     */
    public static double[][] reduceMatrix(double[][] matrix) {

        int row, col, currCol, currRow, nrows, ncols, nonZeroRows = 0, nonZeroColumns = 0;
        double[] rtotal, ctotal;
        double[][] newMatrix;

        nrows = matrix.length;
        ncols = matrix[0].length;
        rtotal = new double[nrows];
        ctotal = new double[ncols];
        for (row = 0; row < nrows; row++) {
            for (col = 0; col < ncols; col++) {
                rtotal[row] += matrix[row][col];
                ctotal[col] += matrix[row][col];
            }
        }
        for (row = 0; row < nrows; row++) {
            if (MLUtils.floatingPointGreaterThan(rtotal[row], 0)) {
                nonZeroRows++;
            }
        }
        for (col = 0; col < ncols; col++) {
            if (MLUtils.floatingPointGreaterThan(ctotal[col], 0)) {
                nonZeroColumns++;
            }
        }
        newMatrix = new double[nonZeroRows][nonZeroColumns];
        currRow = 0;
        for (row = 0; row < nrows; row++) {
            if (MLUtils.floatingPointGreaterThan(rtotal[row], 0)) {
                currCol = 0;
                for (col = 0; col < ncols; col++) {
                    if (MLUtils.floatingPointGreaterThan(ctotal[col], 0)) {
                        newMatrix[currRow][currCol] = matrix[row][col];
                        currCol++;
                    }
                }
                currRow++;
            }
        }
        return newMatrix;
    }

    public double symmetricalUncertainty() {
        return symmetricalUncertainty(entryValues);
    }
    
    /**
     * Calculates the symmetrical uncertainty for base 2.
     * 
     * @param matrix
     *            the contingency table
     * @return the calculated symmetrical uncertainty
     * 
     */
    public static double symmetricalUncertainty(double matrix[][]) {

        double sumForColumn, sumForRow, total = 0, columnEntropy = 0, rowEntropy = 0, entropyConditionedOnRows = 0, infoGain = 0;

        // Compute entropy for columns
        for (int i = 0; i < matrix[0].length; i++) {
            sumForColumn = 0;
            for (int j = 0; j < matrix.length; j++) {
                sumForColumn += matrix[j][i];
            }
            columnEntropy += lnFunc(sumForColumn);
            total += sumForColumn;
        }
        columnEntropy -= lnFunc(total);

        // Compute entropy for rows and conditional entropy
        for (int i = 0; i < matrix.length; i++) {
            sumForRow = 0;
            for (int j = 0; j < matrix[0].length; j++) {
                sumForRow += matrix[i][j];
                entropyConditionedOnRows += lnFunc(matrix[i][j]);
            }
            rowEntropy += lnFunc(sumForRow);
        }
        entropyConditionedOnRows -= rowEntropy;
        rowEntropy -= lnFunc(total);
        infoGain = columnEntropy - entropyConditionedOnRows;
        if (MLUtils.floatingPointEquals(columnEntropy, 0)
                || MLUtils.floatingPointEquals(rowEntropy, 0))
            return 0;
        return 2.0 * (infoGain / (columnEntropy + rowEntropy));
    }

    
    public double tauVal() {
        return tauVal(entryValues);
    }
    
    /**
     * Computes Goodman and Kruskal's tau-value for a contingency table.
     * 
     * @param matrix
     *            the contingency table
     * @return Goodman and Kruskal's tau-value
     */
    public static double tauVal(double[][] matrix) {

        int nrows, ncols, row, col;
        double[] ctotal;
        double maxcol = 0, max, maxtotal = 0, n = 0;

        nrows = matrix.length;
        ncols = matrix[0].length;
        ctotal = new double[ncols];
        for (row = 0; row < nrows; row++) {
            max = 0;
            for (col = 0; col < ncols; col++) {
                if (MLUtils.floatingPointGreaterThan(matrix[row][col], max))
                    max = matrix[row][col];
                ctotal[col] += matrix[row][col];
                n += matrix[row][col];
            }
            maxtotal += max;
        }
        if (MLUtils.floatingPointEquals(n, 0)) {
            return 0;
        }
        maxcol = ctotal[MLUtils.maxIndex(ctotal)];
        return (maxtotal - maxcol) / (n - maxcol);
    }
    
    
    // element probabilities
    public double probOfRow(int row) {
        return laplaceProbOfRow(row, 0);
    }
    
    public double laplaceProbOfRow(int row, double lambda) {
        return (rowTotals[row] + lambda)/(total + numRows*lambda);
    }
    
    public double probOfCol(int col) {
        return laplaceProbOfCol(col, 0);
    }
    
    public double laplaceProbOfCol(int col, double lambda) {
        return (colTotals[col] + lambda)/(total + numCols*lambda);
    }
    
    public double probOfElementGivenRow(int row, int column) {
        return laplaceProbOfElementGivenRow(row, column, 0);
    }
    
    public double laplaceProbOfElementGivenRow(int row, int column, double lambda) {
        return (entryValues[row][column] + lambda)/(rowTotals[row] + numCols*lambda);
    }
    
    public double probOfElementGivenColumn(int row, int column) {
        return laplaceProbOfElementGivenColumn(row, column, 0);
    }
    
    public double laplaceProbOfElementGivenColumn(int row, int column, double lambda) {
        return (entryValues[row][column] + lambda) / (colTotals[column] + numRows*lambda);
    }
    
    public double probOfElement(int row, int column) {
        return laplaceProbOfElement(row, column, 0);
    }
    
    public double laplaceProbOfElement(int row, int column, double lambda) {
        return (entryValues[row][column] + lambda) / (total + numRows*numCols*lambda);
    }
}
