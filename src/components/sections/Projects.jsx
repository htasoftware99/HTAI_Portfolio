import React, { useState } from "react";
import styled from "styled-components";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import GitHubIcon from "@mui/icons-material/GitHub";
import OpenInNewRoundedIcon from "@mui/icons-material/OpenInNewRounded";
import { projects } from "../../data/constants";
import ProjectCard from "../cards/ProjectCard";

const Container = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  margin-top: 50px;
  padding: 0px 16px;
  position: relative;
  z-index: 1;
  align-items: center;
`;

const Wrapper = styled.div`
  position: relative;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-direction: column;
  width: 100%;
  max-width: 1100px;
  gap: 12px;
  @media (max-width: 960px) {
    flex-direction: column;
  }
`;
const Title = styled.div`
  font-size: 52px;
  text-align: center;
  font-weight: 600;
  margin-top: 20px;
  color: ${({ theme }) => theme.text_primary};
  @media (max-width: 768px) {
    margin-top: 12px;
    font-size: 32px;
  }
`;
const Desc = styled.div`
  font-size: 18px;
  text-align: center;
  font-weight: 600;
  color: ${({ theme }) => theme.text_secondary};
  @media (max-width: 768px) {
    font-size: 16px;
  }
`;

const ToggleButtonGroup = styled.div`
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  border: 1.5px solid ${({ theme }) => theme.primary};
  color: ${({ theme }) => theme.primary};
  font-size: 16px;
  border-radius: 12px;
  font-weight: 500;
  margin: 22px 0;
  overflow: hidden;
  @media (max-width: 768px) {
    width: 100%;
    flex-wrap: nowrap;
    font-size: 10px;
  }
`;
const ToggleButton = styled.div`
  padding: 8px 18px;
  border-radius: 6px;
  cursor: pointer;
  white-space: nowrap;
  &:hover {
    background: ${({ theme }) => theme.primary + 20};
  }
  @media (max-width: 768px) {
    padding: 6px 5px;
    border-radius: 4px;
    flex: 1 1 0;
    text-align: center;
  }
  @media (max-width: 360px) {
    padding: 6px 3px;
    font-size: 9px;
  }
  ${({ active, theme }) =>
    active &&
    `
  background:  ${theme.primary + 20};
  `}
`;
const Divider = styled.div`
  width: 1.5px;
  background: ${({ theme }) => theme.primary};
`;

const CardContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 28px;
  flex-wrap: wrap;
`;

const ModalBackdrop = styled.div`
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 28px;
  background: rgba(0, 0, 0, 0.76);
  backdrop-filter: blur(8px);

  @media (max-width: 768px) {
    padding: 14px;
    align-items: flex-end;
  }
`;

const DetailPanel = styled.div`
  position: relative;
  width: min(1040px, 100%);
  max-height: 88vh;
  overflow: hidden auto;
  border-radius: 12px;
  background: ${({ theme }) => theme.card};
  box-shadow: 0 28px 90px rgba(0, 0, 0, 0.7);
  border: 1px solid ${({ theme }) => theme.primary + 30};

  @media (max-width: 768px) {
    max-height: 92vh;
    border-radius: 12px 12px 0 0;
  }
`;

const DetailHero = styled.div`
  position: relative;
  min-height: 330px;
  display: flex;
  align-items: flex-end;
  padding: 40px;
  background:
    linear-gradient(90deg, ${({ theme }) => theme.card} 0%, rgba(23, 23, 33, 0.88) 34%, rgba(23, 23, 33, 0.2) 100%),
    linear-gradient(0deg, ${({ theme }) => theme.card} 0%, rgba(23, 23, 33, 0.08) 45%),
    url(${({ image }) => image}) center/cover;

  @media (max-width: 768px) {
    min-height: 260px;
    padding: 28px 22px;
    background:
      linear-gradient(0deg, ${({ theme }) => theme.card} 0%, rgba(23, 23, 33, 0.52) 58%, rgba(23, 23, 33, 0.12) 100%),
      url(${({ image }) => image}) center/cover;
  }
`;

const CloseButton = styled.button`
  position: absolute;
  top: 18px;
  right: 18px;
  z-index: 2;
  width: 42px;
  height: 42px;
  border: 0;
  border-radius: 50%;
  color: ${({ theme }) => theme.text_primary};
  background: rgba(0, 0, 0, 0.62);
  cursor: pointer;
  display: grid;
  place-items: center;
  transition: background 0.2s ease;

  &:hover {
    background: rgba(0, 0, 0, 0.84);
  }
`;

const HeroContent = styled.div`
  width: min(560px, 100%);
`;

const DetailTitle = styled.h3`
  color: ${({ theme }) => theme.text_primary};
  font-size: 38px;
  line-height: 1.18;
  font-weight: 700;
  margin-bottom: 10px;
  overflow-wrap: anywhere;

  @media (max-width: 768px) {
    font-size: 28px;
  }
`;

const MetaRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  color: ${({ theme }) => theme.text_secondary};
  font-size: 14px;
  font-weight: 500;
`;

const MetaPill = styled.span`
  padding: 6px 10px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.08);
  text-transform: capitalize;
`;

const DetailBody = styled.div`
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 30px;
  padding: 30px 40px 40px;

  @media (max-width: 860px) {
    grid-template-columns: 1fr;
    padding: 24px 22px 32px;
  }
`;

const SectionLabel = styled.div`
  color: ${({ theme }) => theme.text_primary};
  font-size: 16px;
  font-weight: 700;
  margin-bottom: 12px;
`;

const DetailDescription = styled.p`
  color: ${({ theme }) => theme.text_secondary};
  font-size: 15px;
  line-height: 1.8;
`;

const TagList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
`;

const DetailTag = styled.span`
  color: ${({ theme }) => theme.primary};
  background: ${({ theme }) => theme.primary + 16};
  border: 1px solid ${({ theme }) => theme.primary + 35};
  border-radius: 8px;
  padding: 7px 11px;
  font-size: 13px;
  font-weight: 600;
  max-width: 100%;
  overflow-wrap: anywhere;
`;

const SidePanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 22px;
`;

const GitHubLink = styled.a`
  width: 100%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 16px;
  border-radius: 8px;
  color: ${({ theme }) => theme.white};
  background: ${({ theme }) => theme.primary};
  text-decoration: none;
  font-weight: 700;
  transition: transform 0.2s ease, filter 0.2s ease;
  text-align: center;

  &:hover {
    transform: translateY(-2px);
    filter: brightness(1.08);
  }
`;

const Members = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const Member = styled.a`
  display: flex;
  align-items: center;
  gap: 10px;
  color: ${({ theme }) => theme.text_secondary};
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
`;

const MemberAvatar = styled.img`
  width: 34px;
  height: 34px;
  border-radius: 50%;
  object-fit: cover;
  border: 2px solid ${({ theme }) => theme.primary + 55};
`;

const Projects = () => {
  const [toggle, setToggle] = useState("all");
  const [selectedProject, setSelectedProject] = useState(null);

  const visibleProjects =
    toggle === "all"
      ? projects
      : projects.filter((item) => item.category === toggle);

  return (
    <Container id="Projects">
      <Wrapper>
        <Title>Projects</Title>
        <Desc
          style={{
            marginBottom: "40px",
          }}
        >
          I have developed a range of projects in machine learning, deep learning, and generative AI.
        </Desc>

        <ToggleButtonGroup>
          <ToggleButton
            active={toggle === "all"}
            onClick={() => setToggle("all")}
          >
            ALL
          </ToggleButton>
          <Divider />
          <ToggleButton
            active={toggle === "mlops"}
            onClick={() => setToggle("mlops")}
          >
            MLOPS
          </ToggleButton>
          <Divider />
          <ToggleButton
            active={toggle === "deep learning"}
            onClick={() => setToggle("deep learning")}
          >
            DEEP LEARNING
          </ToggleButton>
          <Divider />
          <ToggleButton
            active={toggle === "generative ai"}
            onClick={() => setToggle("generative ai")}
          >
            GENERATIVE AI
          </ToggleButton>
          <Divider />
          <ToggleButton
            active={toggle === "agentic ai"}
            onClick={() => setToggle("agentic ai")}
          >
            AGENTIC AI
          </ToggleButton>
        </ToggleButtonGroup>

        <CardContainer>
          {visibleProjects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onOpen={setSelectedProject}
            />
          ))}
        </CardContainer>
      </Wrapper>
      {selectedProject && (
        <ModalBackdrop onClick={() => setSelectedProject(null)}>
          <DetailPanel onClick={(event) => event.stopPropagation()}>
            <DetailHero image={selectedProject.image}>
              <CloseButton
                type="button"
                aria-label="Close project details"
                onClick={() => setSelectedProject(null)}
              >
                <CloseRoundedIcon />
              </CloseButton>
              <HeroContent>
                <DetailTitle>{selectedProject.title}</DetailTitle>
                <MetaRow>
                  <MetaPill>{selectedProject.category}</MetaPill>
                  <MetaPill>{selectedProject.date}</MetaPill>
                </MetaRow>
              </HeroContent>
            </DetailHero>
            <DetailBody>
              <div>
                <SectionLabel>Project Overview</SectionLabel>
                <DetailDescription>{selectedProject.description}</DetailDescription>
              </div>
              <SidePanel>
                <div>
                  <SectionLabel>Technologies</SectionLabel>
                  <TagList>
                    {selectedProject.tags?.map((tag) => (
                      <DetailTag key={tag}>{tag}</DetailTag>
                    ))}
                  </TagList>
                </div>
                {selectedProject.member?.length > 0 && (
                  <div>
                    <SectionLabel>Team</SectionLabel>
                    <Members>
                      {selectedProject.member.map((member) => (
                        <Member
                          key={member.github || member.name}
                          href={member.github}
                          target="_blank"
                          rel="noreferrer"
                        >
                          <MemberAvatar src={member.img} alt={member.name} />
                          {member.name}
                        </Member>
                      ))}
                    </Members>
                  </div>
                )}
                <GitHubLink
                  href={selectedProject.github}
                  target="_blank"
                  rel="noreferrer"
                >
                  <GitHubIcon fontSize="small" />
                  GitHub Repository
                  <OpenInNewRoundedIcon fontSize="small" />
                </GitHubLink>
              </SidePanel>
            </DetailBody>
          </DetailPanel>
        </ModalBackdrop>
      )}
    </Container>
  );
};

export default Projects;
