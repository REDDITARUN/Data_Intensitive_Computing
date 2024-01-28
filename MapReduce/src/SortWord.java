import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class SortWord {
    public static void main(String[] args) {
        if (args.length != 1) {
            
            System.exit(1);
        }

        String fileName = args[0];

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            Map<String, Integer> wordCounts = new HashMap<>();

            while ((line = br.readLine()) != null) {
                String[] wordCount = line.split("\t");
                if (wordCount.length == 2) {
                    String word = wordCount[0];
                    int count = Integer.parseInt(wordCount[1].trim());
                    wordCounts.put(word, count);
                }
            }

            List<Map.Entry<String, Integer>> sortedWordCounts = new ArrayList<>(wordCounts.entrySet());
            sortedWordCounts.sort(Comparator.comparing(Map.Entry::getValue, Comparator.reverseOrder()));

            List<Map.Entry<String, Integer>> top25 = sortedWordCounts.subList(0, Math.min(25, sortedWordCounts.size()));
            System.out.println("The top 25 words with highest counts");

            for (Map.Entry<String, Integer> entry : top25) {
                String word = entry.getKey();
                int count = entry.getValue();
                
                System.out.println( word + " - " + count);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
