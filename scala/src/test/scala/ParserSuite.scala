import org.scalatest.FunSuite

/**
  * Created by user on 2017-01-18.
  */
class ParserSuite extends FunSuite {

  test("Simple"){
    val str = "언급 안 되었지만 이런 일련의 과정이 순환되기 위한 직접적인 계열구조는 앞서 말한 3사와는 그 급이 비교할수 없을 정도입니다. 아주 수많은 이유중 몇가지만 앝은 지식으로 주절거려봤네요. 하지만 대충 이런 이유만으로도 르삼이나 GM, 쌍용등은 현기차를 잡지 못합니다. 현실이 그렇습니다. 최근 현대나 기아가 내수 MS가 떨어졌던데 그건 수입차 점유율이 높아 그런거지 이 3사에 밀린건 아닙니다. 최근 가장 두각을 나타내는 수입차 업체로 폭스바겐을 주목하는데요. 폭스바겐도 서비스는 개판 오분전입니다. 북미에선 폭스바겐 제품을 레몬카라고 잔고장으론 최악으로 꼽고 있습니다. 이제 좀 더 나은 서비스, 기업혁신, 우수한 제품 개발, 그리고 시대를 넘어서는 선구안등이 회사의 흥망을 좌우하겠죠. 강조한데로 위3 사는 절 대 아닙니다."
    val str2 = "철광석을 가공하고(현대제철) 엔진과 미션을 개발해서(현대 파워텍) 자체 디자인팀과 남양연구소의 우수한 연구원들이 차량을 만듭니다."
    val tree = kaistparser.parse("폭스바겐도 서비스는 개판 오분전입니다.")
    print(tree)
    print(tree.root.allStr)
    val tree2 = kaistparser.parse(str2)
    print(tree2.root.allStr)
  }

}
