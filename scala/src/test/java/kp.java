
import kaist.cilab.parser.berkeleyadaptation.BerkeleyParserWrapper;
import kaist.cilab.tripleextractor.util.Configuration;


public class kp {

    /**
     * @param args
     */
    public static void main(String[] args) {
        // TODO Auto-generated method stub

        String sentence = "영등포구청역에 있는 맛집 좀 알려주세요.";
        String s2 = "폭스바겐도 서비스는 개판 오분전입니다~.";
        String parserPath = "./models/parser/KorGrammar_BerkF_FIN";
        BerkeleyParserWrapper bpw	= new BerkeleyParserWrapper(parserPath); // Path for parser model

        //parse result
        //BerkeleyParserWrapper bpw = new BerkeleyParserWrapper(Configuration.parserModel);
        String str = sentence;
        //1. parse the sentence
        String result = bpw.parse(s2);
        //2. convert PSG-> DG
        System.out.println(result);
    }

}